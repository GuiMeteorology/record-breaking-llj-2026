#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: skewt_satellite_images.py
Description: Plot combined Skew-T diagram soundings, hodographs, and GOES 
             satellite images for atmospheric analysis.

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

import sys
import string
import datetime as dt
from pathlib import Path
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units


# ============================================================
# GLOBAL CONFIGURATION & GENERALIZATION PARAMETERS
# ============================================================

# Directory Paths
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
CACHE_DIR = "data/cache"

# Input/Output Files
DEFAULT_SOUNDING_CSV = os.path.join(INPUT_DIR, "input_soundings.csv")
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "skewt_satellite_panel.png")

# Target Station Metadata
STATION_NAME = "station_name"

# Selected Sounding Dates for Skew-T Plotting
SELECTED_SOUNDING_DATES = [
    'YYYY-MM-DD HH:MM:SS'
]

# GOES Satellite Image Configuration
SATELLITE_DATETIME = "YYYY-MM-DD HH:MM"
SATELLITE_NAME = "G19" # goes satellite
SATELLITE_PRODUCT = "ABI-L2-CMIPF" # product
SATELLITE_BAND = 13 # band

# Geographical Bounding Box (Lat/Lon extent for Satellite Plot)
SAT_LAT_RANGE = (-36.0, -16.0)   # (lat_min, lat_max)
SAT_LON_RANGE = (-68.0, -48.0)   # (lon_min, lon_max)

# Target Meteorological Stations to Display on Map: (longitude, latitude, "Name")
MAP_STATIONS = [
    (-54.4850, -25.6003, "Foz do Iguaçu"),
    (-53.7000, -29.7167, "Santa Maria"),
    (-57.6667, -19.0000, "Corumbá"),
]

# Country & State Labels for Map Overlay: (longitude, latitude, "Label")
MAP_GEO_LABELS = [
    (Lon, Lay, "Label")
]

# AWS S3 Buckets for GOES Satellites
GOES_BUCKETS = {
    "G16": "noaa-goes16",
    "G17": "noaa-goes17",
    "G18": "noaa-goes18",
    "G19": "noaa-goes19"
}


# ============================================================
# 1. SKEW-T & THERMODYNAMICS DATA PREPARATION
# ============================================================

def load_sounding(file_path: str, target_dates=None) -> list:
    """
    Loads sounding data from a CSV file and filters valid thermodynamic 
    and wind profiles for specified timestamps.

    Parameters:
        file_path (str or Path): Path to the CSV file containing sounding data.
        target_dates (list or str, optional): Single datetime string or list of dates to extract.

    Returns:
        list: A list of tuples containing (thermo_df, wind_df) for each valid timestamp.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_cols = ["press_hpa", "temp_c", "dpdp_c", "wdir", "wspd_ms", "datetime"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in file: {missing_cols}")

    df["datetime"] = pd.to_datetime(df["datetime"])

    if target_dates is None:
        target_dates_list = df["datetime"].unique()
    elif isinstance(target_dates, (list, tuple, np.ndarray, pd.Series)):
        target_dates_list = [pd.to_datetime(d) for d in target_dates]
    else:
        target_dates_list = [pd.to_datetime(target_dates)]

    soundings = []

    for d in target_dates_list:
        sounding = df[df["datetime"] == d].copy()

        if sounding.empty:
            print(f"Warning: No sounding found for {d}. Skipping...")
            continue

        thermo = sounding.dropna(subset=["press_hpa", "temp_c", "dpdp_c"]).copy()
        thermo["dewpoint_c"] = thermo["temp_c"] - thermo["dpdp_c"]
        thermo = thermo.drop_duplicates("press_hpa").sort_values("press_hpa", ascending=False).reset_index(drop=True)

        wind = sounding.dropna(subset=["press_hpa", "wdir", "wspd_ms"]).copy()
        wind = wind.drop_duplicates("press_hpa").sort_values("press_hpa", ascending=False).reset_index(drop=True)

        if len(wind) < 2:
            print(f"Warning: Insufficient wind levels for hodograph on {d}. Skipping...")
            continue

        metadata = {
            "datetime": d,
            "lat": sounding["lat"].iloc[0] if "lat" in sounding else "",
            "lon": sounding["lon"].iloc[0] if "lon" in sounding else "",
            "station": sounding["id"].iloc[0] if "id" in sounding else ""
        }
        thermo.attrs.update(metadata)
        wind.attrs.update(metadata)

        soundings.append((thermo, wind))

    if not soundings:
        raise ValueError("No valid soundings were processed.")

    return soundings


def compute_uv_components(wind_df: pd.DataFrame):
    """
    Computes horizontal wind components (U and V) in knots and m/s.

    Parameters:
        wind_df (pd.DataFrame): DataFrame containing 'wdir' (degrees) and 'wspd_ms' (m/s).

    Returns:
        tuple: (pressure, u_kt, v_kt, u_ms, v_ms) as Pint Quantities with MetPy units.
    """
    direction = np.deg2rad(wind_df["wdir"].values)
    speed = wind_df["wspd_ms"].values * units("m/s")

    u_kt = (-speed * np.sin(direction)).to("knots")
    v_kt = (-speed * np.cos(direction)).to("knots")
    u_ms = (-speed * np.sin(direction)).to("m/s")
    v_ms = (-speed * np.cos(direction)).to("m/s")
    pressure = wind_df["press_hpa"].values * units.hPa

    return pressure, u_kt, v_kt, u_ms, v_ms


def calculate_thermo_indices(thermo_df: pd.DataFrame, wind_df: pd.DataFrame) -> dict:
    """
    Calculates thermodynamic and kinematic stability indices using MetPy.

    Parameters:
        thermo_df (pd.DataFrame): Thermodynamic profiles (pressure, temperature, dewpoint).
        wind_df (pd.DataFrame): Wind profile data (direction, speed).

    Returns:
        dict: A dictionary containing key stability parameters (CAPE, CIN, LCL, Shear, SRH, etc.).
    """
    p = thermo_df["press_hpa"].values * units.hPa
    t = thermo_df["temp_c"].values * units.degC
    td = thermo_df["dewpoint_c"].values * units.degC
    p_wind, u_kt, v_kt, u_ms, v_ms = compute_uv_components(wind_df)

    # Surface-Based Parcel Profile and CAPE/CIN
    sb_parcel = mpcalc.parcel_profile(p, t[0], td[0]).to("degC")
    sbcape, sbcin = mpcalc.cape_cin(p, t, td, sb_parcel)

    # Mixed-Layer CAPE/CIN (lowest 100 hPa)
    try:
        _, ml_t, ml_td = mpcalc.mixed_layer(p, t, td, depth=100 * units.hPa)
        ml_parcel = mpcalc.parcel_profile(p, ml_t, ml_td).to("degC")
        mlcape, mlcin = mpcalc.cape_cin(p, t, td, ml_parcel)
    except Exception:
        mlcape, mlcin = np.nan * units("J/kg"), np.nan * units("J/kg")

    # Most Unstable CAPE/CIN
    try:
        _, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, t, td)
        mu_parcel = mpcalc.parcel_profile(p, mu_t, mu_td).to("degC")
        mucape, mucin = mpcalc.cape_cin(p, t, td, mu_parcel)
    except Exception:
        mucape, mucin = np.nan * units("J/kg"), np.nan * units("J/kg")

    # Critical Heights (LCL, LFC, EL)
    lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])

    try:
        lfc_p, _ = mpcalc.lfc(p, t, td)
        el_p, _ = mpcalc.el(p, t, td, sb_parcel)
    except Exception:
        lfc_p = np.nan * units.hPa
        el_p = np.nan * units.hPa

    # Precipitable Water
    try:
        pwat = mpcalc.precipitable_water(p, td)
    except Exception:
        pwat = np.nan * units.mm

    height = mpcalc.pressure_to_height_std(p_wind).to("km")

    # Bulk Wind Shear (0-6 km)
    try:
        shear06 = mpcalc.bulk_shear(p_wind, u_kt, v_kt, height=height, depth=6 * units.km)
        shear06_mag = np.sqrt(shear06[0]**2 + shear06[1]**2)
    except Exception:
        shear06_mag = np.nan * units.knots

    # Storm-Relative Helicity (0-1 km and 0-3 km)
    try:
        srh01 = mpcalc.storm_relative_helicity(height, u_kt, v_kt, depth=1 * units.km)[0]
        srh03 = mpcalc.storm_relative_helicity(height, u_kt, v_kt, depth=3 * units.km)[0]
    except Exception:
        srh01 = np.nan * units.meter**2 / units.second**2
        srh03 = np.nan * units.meter**2 / units.second**2

    indices = {
        "sbcape": sbcape, "sbcin": sbcin,
        "mlcape": mlcape, "mlcin": mlcin,
        "mucape": mucape, "mucin": mucin,
        "lcl": lcl_p, "lfc": lfc_p, "el": el_p, "pwat": pwat,
        "shear06": shear06_mag, "srh01": srh01, "srh03": srh03,
        "p": p, "T": t, "Td": td, "parcel": sb_parcel, "p_wind": p_wind,
        "u_kt": u_kt, "v_kt": v_kt, "u_ms": u_ms, "v_ms": v_ms, "height": height
    }
    return indices


def format_indices_text(idx: dict) -> str:
    """
    Formats calculated thermodynamic indices into a structured text string for plotting.
    """
    def fmt(x):
        try:
            return f"{x.m:.0f}"
        except Exception:
            return "NA"

    return (
        f"SBCAPE: {fmt(idx['sbcape'])} J/kg | SBCIN: {fmt(idx['sbcin'])} J/kg\n"
        f"MLCAPE: {fmt(idx['mlcape'])} J/kg | MLCIN: {fmt(idx['mlcin'])} J/kg\n"
        f"MUCAPE: {fmt(idx['mucape'])} J/kg | MUCIN: {fmt(idx['mucin'])} J/kg\n"
        f"----------------------------------------\n"
        f"LCL: {fmt(idx['lcl'])} hPa | LFC: {fmt(idx['lfc'])} hPa | EL: {fmt(idx['el'])} hPa\n"
        f"PWAT: {fmt(idx['pwat'])} mm | Shear 0-6km: {fmt(idx['shear06'])} kt\n"
        f"SRH 0-1 km: {fmt(idx['srh01'])} m²/s² | SRH 0-3 km: {fmt(idx['srh03'])} m²/s²"
    )


def plot_skewt_panel(fig: plt.Figure, rect: tuple, thermo_df: pd.DataFrame, 
                     wind_df: pd.DataFrame, indices: dict, panel_label: str = "a)"):
    """
    Plots a complete Skew-T Log-P diagram with thermodynamic profiles, wind barbs, 
    an inset hodograph, and stability indices text block.
    """
    idx = indices
    skew = SkewT(fig, rotation=45, rect=rect)

    # Temperature, Dewpoint, and Parcel Profile Lines
    skew.plot(idx["p"], idx["T"], "r", linewidth=2, label="Temperature")
    skew.plot(idx["p"], idx["Td"], "g", linewidth=2, label="Dewpoint")
    skew.plot(idx["p"], idx["parcel"], "k--", linewidth=1.5, label="Parcel")
    skew.shade_cape(idx["p"], idx["T"], idx["parcel"], alpha=0.2, color='red')

    # Plot Wind Barbs along Pressure Levels
    wind_mask = (idx["p_wind"] <= 1000 * units.hPa) & (idx["p_wind"] >= 100 * units.hPa)
    skew.plot_barbs(
        idx["p_wind"][wind_mask], idx["u_kt"][wind_mask], idx["v_kt"][wind_mask],
        length=6, xloc=1.0, color="black", flip_barb=True
    )

    def get_parcel_temp_at_level(target_p, p_array, parcel_array):
        return np.interp(target_p.m, p_array.m[::-1], parcel_array.m[::-1]) * units.degC

    # Mark LCL, LFC, and EL levels on the Skew-T diagram
    if not np.isnan(idx["lcl"].m):
        t_lcl = get_parcel_temp_at_level(idx["lcl"], idx["p"], idx["parcel"])
        skew.plot(idx["lcl"], t_lcl, marker="_", color="green", markersize=20, markeredgewidth=2.5)
        skew.ax.text(t_lcl.m + 2.5, idx["lcl"].m, "LCL", color="green", fontweight="bold", fontsize=8, verticalalignment="center")

    if not np.isnan(idx["lfc"].m):
        t_lfc = get_parcel_temp_at_level(idx["lfc"], idx["p"], idx["parcel"])
        skew.plot(idx["lfc"], t_lfc, marker="_", color="gold", markersize=20, markeredgewidth=2.5)
        skew.ax.text(t_lfc.m + 2.5, idx["lfc"].m, "LFC", color="gold", fontweight="bold", fontsize=8, verticalalignment="center")

    if not np.isnan(idx["el"].m):
        t_el = get_parcel_temp_at_level(idx["el"], idx["p"], idx["parcel"])
        skew.plot(idx["el"], t_el, marker="_", color="purple", markersize=20, markeredgewidth=2.5)
        skew.ax.text(t_el.m + 2.5, idx["el"].m, "EL", color="purple", fontweight="bold", fontsize=8, verticalalignment="center")

    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 45)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")

    skew.plot_dry_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_moist_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_mixing_lines(linewidth=0.5, alpha=0.5)

    # Panel Subplot Letter Label
    skew.ax.text(
        0.02, 0.95, panel_label, transform=skew.ax.transAxes,
        fontsize=16, fontweight="bold", verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2)
    )

    skew.ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.01, 0.88), framealpha=0.85)

    dt_attr = thermo_df.attrs.get("datetime", "")
    dt_str = dt_attr.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt_attr, pd.Timestamp) else str(dt_attr)
    skew.ax.set_title(f"{STATION_NAME} | {dt_str} UTC", fontsize=12, fontweight='bold', loc="left", pad=8)

    # Stability Indices Info Box
    skew.ax.text(
        0.02, 0.02, format_indices_text(indices), transform=skew.ax.transAxes,
        fontsize=9, verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3")
    )

    # --------------------------------------------------------
    # INSET HODOGRAPH SETUP
    # --------------------------------------------------------
    ax_hodo = inset_axes(
        skew.ax, width="100%", height="100%",
        bbox_to_anchor=(0.65, 0.58, 0.30, 0.30),
        bbox_transform=skew.ax.transAxes, loc="upper right"
    )

    # Colorbar Attached Above the Hodograph
    cax = inset_axes(
        ax_hodo, width="100%", height="4%", loc="lower center",
        bbox_to_anchor=(0.0, 1.08, 1.0, 1.0),
        bbox_transform=ax_hodo.transAxes, borderpad=0
    )

    # Wind Height Filtering (0-12 km)
    h_mask = (indices["height"] >= 0 * units.km) & (indices["height"] <= 12 * units.km)
    h_u, h_v, h_alt = indices["u_kt"][h_mask], indices["v_kt"][h_mask], indices["height"][h_mask]

    grid_range = 120   # Rings drawn up to 120 kt
    axis_range = 125   # Plot axis box set to 125 kt
    step_kt = 20

    hodo = Hodograph(ax_hodo, component_range=axis_range)
    hodo.add_grid(increment=step_kt, color="gray", linestyle="--", linewidth=0.6)

    ax_hodo.set_xlim(-axis_range, axis_range)
    ax_hodo.set_ylim(-axis_range, axis_range)

    # Remove outer ticks and numeric labels
    ax_hodo.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False
    )

    # Draw solid black bounding frame for Hodograph
    for spine in ax_hodo.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # Add numeric ring labels (20, 40, 60, 80, 100, 120)
    for r in range(step_kt, grid_range + 1, step_kt):
        ax_hodo.text(
            0, r, f" {r}", color="gray", fontsize=6, 
            verticalalignment="center", horizontalalignment="left",
            alpha=0.8, fontweight="bold"
        )

    # Wind speed unit label
    ax_hodo.text(
        0.03, 0.93, "kt", transform=ax_hodo.transAxes,
        fontsize=7, fontweight="bold", color="gray",
        ha="left", va="top"
    )

    # Colormap by Height Layers
    cmap_hodo = ListedColormap(["darkred", "darkgreen", "orange", "royalblue"])
    bounds_hodo = [0, 3, 6, 9, 12]
    norm_hodo = BoundaryNorm(bounds_hodo, cmap_hodo.N)

    points = hodo.plot_colormapped(h_u, h_v, h_alt, cmap=cmap_hodo, norm=norm_hodo, linewidth=1.8)

    # Height Colorbar
    cb = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=bounds_hodo, extend="max")
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.ax.xaxis.set_ticks_position('top')
    cb.set_label("Height (km)", fontsize=7, labelpad=2, loc="center")
    cb.ax.xaxis.set_label_position('top')

    ax_hodo.set_facecolor("white")


# ============================================================
# 2. GOES SATELLITE PROCESSING & PLOTTING
# ============================================================

def download_goes_image(date_str: str, sat: str, product: str, band: int, output_dir: str) -> str:
    """
    Downloads the closest matching GOES NetCDF file from NOAA Amazon S3 public bucket.

    Parameters:
        date_str (str): Target datetime string ("YYYY-MM-DD HH:MM").
        sat (str): Satellite identifier key (e.g., "G19").
        product (str): NOAA GOES product name (e.g., "ABI-L2-CMIPF").
        band (int): Satellite channel band number.
        output_dir (str): Directory where downloaded files are saved.

    Returns:
        str: Local path to the downloaded NetCDF file.
    """
    import s3fs
    bucket = GOES_BUCKETS[sat]
    target_dt = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    day_of_year = target_dt.timetuple().tm_yday
    prefix = f"{bucket}/{product}/{target_dt.year}/{day_of_year:03d}/{target_dt.hour:02d}/"

    fs = s3fs.S3FileSystem(anon=True)
    files = fs.ls(prefix)
    band_suffix = f"C{band:02d}_{sat}"
    candidates = [f for f in files if band_suffix in f]

    def extract_file_minute(path):
        filename = os.path.basename(path)
        start_field = [p for p in filename.split("_") if p.startswith("s")][0]
        return int(start_field[8:10]) * 60 + int(start_field[10:12])

    target_minute = target_dt.hour * 60 + target_dt.minute
    candidates.sort(key=lambda c: abs(extract_file_minute(c) - target_minute))
    selected_file = candidates[0]

    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, os.path.basename(selected_file))

    if os.path.exists(local_path):
        print(f"[cache] Using cached satellite file: {local_path}")
        return local_path

    print(f"[download] Downloading satellite image s3://{selected_file} ...")
    fs.get(selected_file, local_path)
    return local_path


def calculate_satellite_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates latitude and longitude coordinates from GOES ABI fixed grid projection.
    """
    x, y = np.meshgrid(ds.x, ds.y)
    proj = ds.goes_imager_projection
    r_eq, r_pol = proj.attrs["semi_major_axis"], proj.attrs["semi_minor_axis"]
    l_0 = proj.attrs["longitude_of_projection_origin"] * (np.pi / 180)
    h_height = r_eq + proj.attrs["perspective_point_height"]

    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * h_height * np.cos(x) * np.cos(y)
    c = h_height**2 - r_eq**2

    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        s_x = r_s * np.cos(x) * np.cos(y)
        s_y = -r_s * np.sin(x)
        s_z = r_s * np.cos(x) * np.sin(y)
        lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((h_height - s_x)**2 + s_y**2))) * (180 / np.pi)
        lon = (l_0 - np.arctan(s_y / (h_height - s_x))) * (180 / np.pi)

    ds = ds.assign_coords({"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)})
    return ds


def get_xy_from_latlon(ds: xr.Dataset, lats: tuple, lons: tuple) -> tuple:
    """
    Retrieves pixel X/Y coordinate slice boundaries corresponding to geographic Lat/Lon ranges.
    """
    mask = (ds.lat.data >= lats[0]) & (ds.lat.data <= lats[1]) & (ds.lon.data >= lons[0]) & (ds.lon.data <= lons[1])
    x_mesh, y_mesh = np.meshgrid(ds.x.data, ds.y.data)
    x_sel, y_sel = x_mesh[mask], y_mesh[mask]
    return ((min(x_sel), max(x_sel)), (min(y_sel), max(y_sel)))


def create_enhanced_ir_colormap():
    """
    Generates a custom enhanced Infrared (IR) brightness temperature colormap.
    """
    vmin, vmax = -110, 40
    color_stops = [
        (-110, "#000000"), (-90, "#3b0000"), (-80, "#FF0000"),
        (-70, "#FFA500"), (-60, "#FFFF00"), (-50, "#00C000"),
        (-40, "#00FFFF"), (-32, "#0000CD"), (-32 + 1e-3, "#FFFFFF"),
        (0, "#4d4d4d"), (40, "#000000")
    ]
    fracs = [(t - vmin) / (vmax - vmin) for t, _ in color_stops]
    colors = [c for _, c in color_stops]
    cmap = mcolors.LinearSegmentedColormap.from_list("enhanced_ir_smooth", list(zip(fracs, colors)))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def plot_satellite_panel(fig: plt.Figure, rect: tuple, panel_label: str = "f)"):
    """
    Plots the GOES IR satellite imagery overlaid with coastlines, borders, 
    meteorological stations, and geographic labels using Cartopy.
    """
    local_file = download_goes_image(
        SATELLITE_DATETIME, SATELLITE_NAME, SATELLITE_PRODUCT, SATELLITE_BAND, CACHE_DIR
    )
    ds = xr.open_dataset(local_file)
    ds = calculate_satellite_latlon(ds)

    (x1, x2), (y1, y2) = get_xy_from_latlon(ds, SAT_LAT_RANGE, SAT_LON_RANGE)
    subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1))
    subset = subset.assign(CMI_C=subset.CMI - 273.15)

    cmap, norm = create_enhanced_ir_colormap()

    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())

    pcm = ax.pcolormesh(
        subset.lon, subset.lat, subset.CMI_C,
        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto"
    )

    # Cartopy Geographic Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="cyan")
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="cyan")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.7, edgecolor="cyan")
    ax.set_extent([SAT_LON_RANGE[0], SAT_LON_RANGE[1], SAT_LAT_RANGE[0], SAT_LAT_RANGE[1]], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Geographic Text Overlay (Country/State labels)
    for lon_lbl, lat_lbl, txt in MAP_GEO_LABELS:
        txt_obj = ax.text(
            lon_lbl, lat_lbl, txt, color="yellow", fontsize=14, fontweight="bold",
            ha="center", va="center", transform=ccrs.PlateCarree()
        )
        txt_obj.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

    # Plot Target Stations
    for lon_st, lat_st, st_name in MAP_STATIONS:
        ax.plot(lon_st, lat_st, marker="o", color="red", markersize=9, markeredgecolor="white", markeredgewidth=1.2, transform=ccrs.PlateCarree())
        ax.text(
            lon_st + 0.5, lat_st, st_name, color="white", fontsize=12, fontweight="bold",
            va="center", transform=ccrs.PlateCarree(),
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2)
        )

    # Colorbar
    cax = inset_axes(ax, width="3%", height="100%", loc="right", bbox_to_anchor=(0.08, 0, 1, 1), bbox_transform=ax.transAxes)
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label("Brightness Temp (°C)", fontsize=10, fontweight="bold")
    cb.ax.tick_params(labelsize=9)

    # Panel Letter Label
    ax.text(
        0.02, 0.95, panel_label, transform=ax.transAxes,
        fontsize=16, fontweight="bold", verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2)
    )

    ax.set_title(f"GOES-19 IR | {SATELLITE_DATETIME} UTC", fontsize=13, fontweight="bold", loc="left", pad=8)


# ============================================================
# 3. COMBINED PANEL GENERATOR
# ============================================================

def generate_combined_figure(soundings_indices: list, output_filepath: str):
    """
    Generates a combined multi-panel figure consisting of 5 Skew-T diagram panels (a-e)
    and 1 GOES IR Satellite image panel (f).

    Parameters:
        soundings_indices (list): List of tuples containing (thermo_df, wind_df, indices_dict).
        output_filepath (str): Output filename path where the image will be saved.
    """
    ncols = 2
    nrows = 3

    fig_width = 8.0 * ncols
    fig_height = 8.0 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))

    letters = [f"{letra})" for letra in string.ascii_lowercase]

    margin_x = 0.05 / ncols
    margin_y = 0.04 / nrows
    gap_x = 0.07 / ncols
    gap_y = 0.05 / nrows

    w_panel = (1.0 - 2 * margin_x - (ncols - 1) * gap_x) / ncols
    h_panel = (1.0 - 2 * margin_y - (nrows - 1) * gap_y) / nrows

    # Plot Soundings (Panels a to e)
    for idx, (thermo, wind, indices) in enumerate(soundings_indices):
        col = idx % ncols
        row = idx // ncols

        l_panel = margin_x + col * (w_panel + gap_x)
        b_panel = 1.0 - margin_y - (row + 1) * h_panel - row * gap_y

        rect = (l_panel, b_panel, w_panel, h_panel)
        plot_skewt_panel(fig, rect, thermo, wind, indices, panel_label=letters[idx])

    # Plot Satellite Image (Panel f, bottom right corner)
    col = 5 % ncols
    row = 5 // ncols
    l_panel = margin_x + col * (w_panel + gap_x)
    b_panel = 1.0 - margin_y - (row + 1) * h_panel - row * gap_y
    rect_sat = (l_panel, b_panel, w_panel, h_panel)

    plot_satellite_panel(fig, rect_sat, panel_label=letters[5])

    plt.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Combined figure successfully saved to: {output_filepath}")


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

if __name__ == "__main__":
    # Ensure working output and cache directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Command line argument for custom CSV input path
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])
    else:
        input_csv = Path(DEFAULT_SOUNDING_CSV)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input sounding file not found at: {input_csv}")

    # Load soundings and compute stability indices
    soundings_list = load_sounding(input_csv, target_dates=SELECTED_SOUNDING_DATES)

    soundings_with_indices = []
    for thermo_data, wind_data in soundings_list:
        calculated_indices = calculate_thermo_indices(thermo_data, wind_data)
        soundings_with_indices.append((thermo_data, wind_data, calculated_indices))

    # Generate output combined plot
    generate_combined_figure(soundings_with_indices, output_filepath=OUTPUT_FIGURE_PATH)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: llj_statistics.py
Description: Statistical analysis, extreme event identification, persistence 
             calculation, and visualization for Low-Level Jet (LLJ) events.

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Configurations & Generalization Parameters
# ---------------------------------------------------------------------------
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

# Define default stations with metadata: (station_name, file_name, latitude, longitude)
STATIONS_CONFIG = [
    ("City", "City_data.csv", Lat, Lon)
    ]

TARGET_PERCENTILE = 90 # adapt for your percentile
MIN_PERSISTENCE_SOUNDINGS = 4 # adapt for minimum persistence

def format_station_name(name: str, lat: float = None, lon: float = None) -> str:
    """
    Formats the station name with its geographic coordinates if provided.

    Parameters:
        name (str): Name of the meteorological station.
        lat (float, optional): Station latitude in decimal degrees.
        lon (float, optional): Station longitude in decimal degrees.

    Returns:
        str: Formatted station name string (e.g., "Corumbá (19.00°S, 57.67°W)").
    """
    if lat is None or lon is None:
        return name

    lat_str = f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'}"
    lon_str = f"{abs(lon):.2f}°{'W' if lon < 0 else 'E'}"

    return f"{name} ({lat_str}, {lon_str})"


# ---------------------------------------------------------------------------
# 0. Data Pre-filtering & Cleaning
# ---------------------------------------------------------------------------

def filter_strict_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by eliminating duplicates and ensuring a single 
    record per sounding timestamp (selecting the maximum wind speed per launch).

    Parameters:
        df (pd.DataFrame): Raw input sounding DataFrame.

    Returns:
        pd.DataFrame: Sorted DataFrame with unique datetime entries.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_copy = df.copy()

    # Select the profile level with the maximum wind speed for each unique sounding timestamp
    df_filtered = (
        df_copy.sort_values("wspd_ms", ascending=False)
        .groupby("datetime", as_index=False)
        .first()
    )
    df_filtered = df_filtered.sort_values("datetime").reset_index(drop=True)

    return df_filtered


# ---------------------------------------------------------------------------
# 1. Percentile Threshold Calculation
# ---------------------------------------------------------------------------

def calculate_station_percentile_threshold(df: pd.DataFrame, percentile: float = 95.0) -> float:
    """
    Calculates the wind speed threshold for a given percentile, restricted 
    exclusively to observations flagged as valid SALLJ events (is_jbn == True).

    Parameters:
        df (pd.DataFrame): Station sounding DataFrame.
        percentile (float): Percentile value to evaluate (default is 95.0).

    Returns:
        float: Calculated wind speed threshold in m/s, or np.nan if empty.
    """
    if df is None or df.empty or "wspd_ms" not in df.columns:
        return np.nan

    # Restrict strict evaluation to verified SALLJ events
    if "is_jbn" in df.columns:
        df_jbn_true = df[df["is_jbn"] == True]
    else:
        df_jbn_true = df

    wind_series = df_jbn_true["wspd_ms"].dropna()
    if wind_series.empty:
        return np.nan

    return float(np.percentile(wind_series, percentile))


# ---------------------------------------------------------------------------
# 2. Time-Gap Discontinuity Handling
# ---------------------------------------------------------------------------

def insert_nans_for_long_gaps(df: pd.DataFrame, date_col: str = "datetime", max_gap_days: int = 90) -> pd.DataFrame:
    """
    Inserts NaNs at the midpoint of temporal gaps exceeding 'max_gap_days' 
    to break continuous plot lines during extended data missing periods.

    Parameters:
        df (pd.DataFrame): Time-series DataFrame.
        date_col (str): Column name representing datetime values.
        max_gap_days (int): Threshold in days to define a major missing data gap.

    Returns:
        pd.DataFrame: DataFrame containing inserted NaN rows to interrupt line plotting.
    """
    if df is None or df.empty:
        return df

    df_sorted = df.sort_values(by=date_col).copy().reset_index(drop=True)
    time_diff = df_sorted[date_col].diff()
    gaps_mask = time_diff > pd.Timedelta(days=max_gap_days)

    if not gaps_mask.any():
        return df_sorted

    new_nan_rows = []
    for idx in df_sorted[gaps_mask].index:
        prev_date = df_sorted.loc[idx - 1, date_col]
        curr_date = df_sorted.loc[idx, date_col]
        midpoint = prev_date + (curr_date - prev_date) / 2

        nan_row = {col: np.nan for col in df_sorted.columns}
        nan_row[date_col] = midpoint
        new_nan_rows.append(nan_row)

    if new_nan_rows:
        df_nans = pd.DataFrame(new_nan_rows)
        df_final = pd.concat([df_sorted, df_nans], ignore_index=True)
        return df_final.sort_values(by=date_col).reset_index(drop=True)

    return df_sorted


# ---------------------------------------------------------------------------
# 3. LLJ Persistence Identification
# ---------------------------------------------------------------------------

def calculate_sounding_persistence(df_complete: pd.DataFrame, threshold: float, 
                                    min_persistence: int = 4, max_gap_hours: float = 25.0) -> pd.DataFrame:
    """
    Identifies persistent SALLJ events under strict criteria:
    1. Wind speed >= threshold AND flagged as SALLJ (is_jbn == True).
    2. Minimum of 'min_persistence' consecutive soundings fulfilling criteria.
    3. Temporal spacing between consecutive soundings must NOT exceed 'max_gap_hours'.

    Parameters:
        df_complete (pd.DataFrame): Full time-series DataFrame containing all soundings.
        threshold (float): Wind speed percentile threshold (m/s).
        min_persistence (int): Minimum number of consecutive soundings required.
        max_gap_hours (float): Maximum allowable time gap between consecutive observations (hours).

    Returns:
        pd.DataFrame: Summary table of identified persistent events.
    """
    if df_complete is None or df_complete.empty:
        return pd.DataFrame(columns=["start", "end", "n_soundings", "wspd_max"])

    df = df_complete.sort_values("datetime").reset_index(drop=True)

    # Combined criteria: wind speed above threshold and SALLJ present
    if "is_jbn" in df.columns:
        is_above = (df["wspd_ms"] >= threshold) & (df["is_jbn"] == True)
    else:
        is_above = df["wspd_ms"] >= threshold

    # Check for missing/too wide temporal gaps between observations
    hours_diff = df["datetime"].diff().dt.total_seconds() / 3600.0
    has_gap = hours_diff > max_gap_hours

    # Break events on condition changes or temporal gaps
    group_break = (is_above != is_above.shift()) | has_gap
    event_group = group_break.cumsum()

    events = []
    for _, group in df.groupby(event_group):
        idx0 = group.index[0]
        if is_above.loc[idx0] and len(group) >= min_persistence:
            events.append({
                "start": group["datetime"].iloc[0],
                "end": group["datetime"].iloc[-1],
                "n_soundings": len(group),
                "wspd_max": group["wspd_ms"].max(),
            })

    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# 4. Visualization & Plotting Functions
# ---------------------------------------------------------------------------

def mark_extreme_events_red_arrows(ax: plt.Axes, df_true: pd.DataFrame, threshold: float, 
                                    val_col: str = "wspd_ms", date_col: str = "datetime") -> None:
    """
    Annotates individual extreme wind speed events on a Matplotlib axis using red arrows.

    Parameters:
        ax (plt.Axes): Target Matplotlib axis object.
        df_true (pd.DataFrame): DataFrame containing valid SALLJ soundings.
        threshold (float): Wind speed percentile threshold (m/s).
        val_col (str): Column name containing wind speed values.
        date_col (str): Column name containing datetime values.
    """
    if df_true is None or df_true.empty:
        return

    extremes = df_true.dropna(subset=[val_col])
    extremes = extremes[extremes[val_col] >= threshold]

    for _, row in extremes.iterrows():
        ax.annotate(
            "",
            xy=(row[date_col], row[val_col]),
            xytext=(0, 18),
            textcoords="offset points",
            arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="-|>", lw=1.2, shrinkA=0, shrinkB=2)
        )


def plot_complete_sallj_grid(stations_dfs: list, percentile: float = 90.0, min_persistence: int = 4, 
                             bins: list = None, gap_limit_days: int = 365, out_path: str = "sallj_grid.png", 
                             hist_color: str = "#DD8452", ts_color: str = "#4C72B0") -> str:
    """
    Generates a multi-panel grid plot containing wind speed histograms and 
    time-series with extreme and persistent LLJ event markers for multiple stations.

    Parameters:
        stations_dfs (list): List of tuples (station_name, df_complete, lat, lon).
        percentile (float): Target percentile threshold for extreme events.
        min_persistence (int): Minimum consecutive soundings for persistent events.
        bins (list, optional): Histogram wind speed bin boundaries.
        gap_limit_days (int): Threshold in days for inserting missing data breaks.
        out_path (str): File path for saving the generated figure.
        hist_color (str): Hex color code for histogram bars.
        ts_color (str): Hex color code for time-series lines.

    Returns:
        str: Saved output image file path.
    """
    if bins is None:
        bins = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                32, 34, 36, 38, 40, np.inf]

    n_stations = len(stations_dfs)
    fig, axes = plt.subplots(n_stations, 2, figsize=(16, 4.3 * n_stations), tight_layout=True)
    if n_stations == 1:
        axes = np.array([axes])

    for i, station_info in enumerate(stations_dfs):
        if len(station_info) == 2:
            station_name, df_complete = station_info
            lat, lon = None, None
        else:
            station_name, df_complete, lat, lon = station_info

        formatted_name = format_station_name(station_name, lat, lon)

        # Separate full dataset from verified SALLJ profiles
        df_true = df_complete[df_complete["is_jbn"] == True] if "is_jbn" in df_complete.columns else df_complete

        station_threshold = calculate_station_percentile_threshold(df_true, percentile=percentile)
        threshold_str = f"{station_threshold:.1f}" if not np.isnan(station_threshold) else "N/A"
        print(f"---> {station_name} | P{percentile} Threshold (SALLJ Only): {threshold_str} m/s")

        label_hist_letter = f"{chr(97 + 2*i)})"
        label_ts_letter = f"{chr(97 + 2*i + 1)})"

        # -------------------------------------------------------------------
        # COLUMN 0: Wind Speed Histogram (SALLJ Events Only)
        # -------------------------------------------------------------------
        ax_hist = axes[i, 0]
        data = df_true["wspd_ms"].dropna() if (df_true is not None and not df_true.empty) else pd.Series(dtype=float)

        ax_hist.hist(data, bins=bins, color=hist_color, edgecolor="black", alpha=0.85)
        ax_hist.set_title(f"Histogram SALLJ (n={len(data)})")
        ax_hist.set_xlabel("Wind Speed (m/s)")
        ax_hist.set_ylabel("Number of SALLJ Occurrences")

        if not data.empty:
            mean_val = data.mean()
            std_val = data.std()

            # Percentile line (Red)
            if not np.isnan(station_threshold):
                ax_hist.axvline(station_threshold, color="red", linestyle="--", linewidth=1.5)

            # Mean line (Purple)
            ax_hist.axvline(mean_val, color="purple", linestyle="--", linewidth=1.5)

            # Build custom legend entries
            handle_percentile = Line2D([], [], color="red", linestyle="--", linewidth=1.5,
                                       label=f"P{percentile} = {threshold_str} m/s")
            handle_mean = Line2D([], [], color="purple", linestyle="--", linewidth=1.5,
                                 label=f"Mean = {mean_val:.1f} m/s")
            handle_std = Line2D([], [], color="none", linestyle="None",
                                label=f"Std Dev = ± {std_val:.1f} m/s")

            ax_hist.legend(handles=[handle_percentile, handle_mean, handle_std],
                            fontsize=9.5, loc="upper right")

        ax_hist.set_ylim(0, 300)
        ax_hist.set_yticks(np.arange(0, 301, 50))

        ax_hist.set_xlim(8, 42)
        ax_hist.set_xticks(bins[:-1])
        bin_labels = [str(b) for b in bins[:-1]]
        bin_labels[-1] = f"{bins[-2]}+"
        ax_hist.set_xticklabels(bin_labels, fontsize=7.5)
        ax_hist.grid(axis="y", linestyle="--", alpha=0.5)

        ax_hist.text(0.02, 0.98, label_hist_letter, transform=ax_hist.transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        ax_hist.text(0.98, 0.68, formatted_name, transform=ax_hist.transAxes,
                     fontsize=11, fontweight="bold", ha="right", va="top")

        # -------------------------------------------------------------------
        # COLUMN 1: SALLJ Time Series & Extreme Event Annotations
        # -------------------------------------------------------------------
        ax_ts = axes[i, 1]
        n_extremes = 0
        n_persistence = 0

        if df_complete is not None and not df_complete.empty and not np.isnan(station_threshold):
            df_nw_complete = df_complete.sort_values("datetime").drop_duplicates("datetime").copy()
            df_nw_true = df_true.sort_values("datetime").drop_duplicates("datetime").copy()

            n_extremes = (df_nw_true["wspd_ms"] >= station_threshold).sum()

            # Calculate persistence events using complete dataset
            persistence_events = calculate_sounding_persistence(
                df_nw_complete, threshold=station_threshold, min_persistence=min_persistence
            )
            n_persistence = len(persistence_events)

            # Insert NaNs for plotting continuous lines cleanly
            df_plot = insert_nans_for_long_gaps(df_nw_true, date_col="datetime", max_gap_days=gap_limit_days)

            # 1. Plot SALLJ wind speed time series
            ax_ts.plot(df_plot["datetime"], df_plot["wspd_ms"], color=ts_color, linewidth=0.8, zorder=3)

            # 2. Mark all extreme events with RED ARROWS
            mark_extreme_events_red_arrows(ax_ts, df_nw_true, threshold=station_threshold)

            # 3. Annotate PERSISTENCE events with BLACK ARROW + DATE
            for _, ev in persistence_events.iterrows():
                date_str = ev["start"].strftime("%d/%m/%y")
                max_speed_event = ev["wspd_max"]

                ax_ts.annotate(
                    date_str,
                    xy=(ev["start"], max_speed_event),
                    xytext=(ev["start"], max_speed_event + 7),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    color="black",
                    arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="-|>", lw=1.3, shrinkB=2),
                    zorder=5,
                    clip_on=False
                )

        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        plt.setp(ax_ts.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        ax_ts.set_ylim(8, 53)
        ax_ts.set_yticks(np.arange(8, 59, 10))
        ax_ts.set_ylabel("Wind Speed (m/s)")

        ax_ts.set_title(f"Time series SALLJ and Extreme Events (P{percentile} = {threshold_str} m/s)", pad=15)

        # Threshold reference horizontal line
        if not np.isnan(station_threshold):
            ax_ts.axhline(station_threshold, color="red", linestyle="--", linewidth=1.5, zorder=4,
                          label=f"P{percentile} Threshold ({threshold_str} m/s)")

        ax_ts.grid(axis="y", linestyle="--", alpha=0.3)

        ax_ts.text(0.02, 0.98, label_ts_letter, transform=ax_ts.transAxes,
                   fontsize=12, fontweight="bold", va="top", ha="left")

        ax_ts.text(0.50, 0.95, formatted_name, transform=ax_ts.transAxes,
                   fontsize=12, fontweight="bold", ha="center", va="top",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

        # Time series legend elements
        handle_red_arrow = Line2D([], [], color="red", marker="v", linestyle="None",
                                   markersize=6, label=f"≥ P{percentile} [{threshold_str} m/s] (n={n_extremes})   |")

        handle_black_arrow = Line2D([], [], color="black", marker="v", linestyle="None",
                                     markersize=6, label=f"Persistence (≥{min_persistence} soundings) (n={n_persistence})")

        leg_ts = ax_ts.legend(
            handles=[handle_red_arrow, handle_black_arrow],
            loc="upper center",
            bbox_to_anchor=(0.55, 0.88),
            ncol=2,
            frameon=True,
            fontsize=10,
            handletextpad=0.4,
            columnspacing=1.2,
            borderpad=0.25
        )
        leg_ts.get_frame().set_facecolor("white")
        leg_ts.get_frame().set_alpha(0.85)
        leg_ts.get_frame().set_linewidth(0)

    fig.suptitle(f"SALLJ and Persistence (Station P{percentile} | 2007 - 2026)",
                 fontsize=16, fontweight="bold", y=1.01)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_station_boxplot(stations_dict: dict, out_path: str = "boxplot.png") -> None:
    """
    Generates and saves a comparative boxplot of SALLJ wind speeds across stations.

    Parameters:
        stations_dict (dict): Dictionary mapping station names to their respective DataFrames.
        out_path (str): Output image file path for saving the boxplot.
    """
    data = []
    labels = []

    for name, df in stations_dict.items():
        if df is None or df.empty:
            continue
        df_jbn = df[df['is_jbn'] == True]['wspd_ms'].dropna() if 'is_jbn' in df.columns else df['wspd_ms'].dropna()
        data.append(df_jbn)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(7, 6))
    positions = list(range(1, len(data) + 1))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=2)
    )

    colors = ["tab:orange", "tab:green", "tab:purple", "tab:blue", "tab:red"]

    for box, color in zip(bp["boxes"], colors[:len(data)]):
        box.set(facecolor=color, edgecolor="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(8, 44, 4))
    ax.set_ylabel("Wind Speed (m s$^{-1}$)")
    ax.set_xlabel("Stations")
    ax.set_title("SALLJ Boxplot (2007 - 2026)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Dataset Aggregation & CSV Export
# ---------------------------------------------------------------------------

def generate_monthly_extremes_dataset(stations_dict: dict, percentile: float = 95.0, 
                                      output_csv: str = "monthly_extremes.csv") -> pd.DataFrame:
    """
    Aggregates extreme SALLJ events into monthly frequency totals per station.

    Parameters:
        stations_dict (dict): Dictionary of station names and DataFrames.
        percentile (float): Percentile threshold for extreme event classification.
        output_csv (str): Path to save exported CSV dataset.

    Returns:
        pd.DataFrame: Summary table with monthly counts of extreme events per station.
    """
    station_summaries = []

    for station_name, df in stations_dict.items():
        if df is None or df.empty:
            continue

        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])

        if 'is_jbn' in df_copy.columns:
            df_copy = df_copy[df_copy['is_jbn'] == True]

        df_filtered = df_copy.dropna(subset=['wspd_ms'])
        if df_filtered.empty:
            continue

        threshold = np.percentile(df_filtered['wspd_ms'], percentile)
        df_extremes = df_filtered[df_filtered['wspd_ms'] >= threshold].copy()

        df_extremes['month_year'] = df_extremes['datetime'].dt.to_period('M')
        counts = df_extremes.groupby('month_year').size().reset_index(name=station_name)
        station_summaries.append(counts)

    if not station_summaries:
        print("No extreme events found across datasets.")
        return pd.DataFrame()

    df_merged = station_summaries[0]
    for df_c in station_summaries[1:]:
        df_merged = pd.merge(df_merged, df_c, on='month_year', how='outer')

    station_cols = [col for col in df_merged.columns if col != 'month_year']
    df_merged[station_cols] = df_merged[station_cols].fillna(0).astype(int)
    df_merged = df_merged.sort_values('month_year').reset_index(drop=True)

    df_merged['total_events'] = df_merged[station_cols].sum(axis=1)
    df_summary = df_merged[df_merged['total_events'] > 0].copy()
    df_summary['month/year'] = df_summary['month_year'].dt.strftime('%m/%Y')

    final_cols = ['month/year'] + station_cols + ['total_events']
    df_summary = df_summary[final_cols]

    df_summary.to_csv(output_csv, index=False)
    print(f"Monthly dataset exported to '{output_csv}' containing {len(df_summary)} recorded months.")

    return df_summary


def generate_annual_extremes_dataset(stations_dict: dict, percentile: float = 95.0, 
                                     output_csv: str = "annual_extremes.csv") -> pd.DataFrame:
    """
    Aggregates extreme SALLJ events into annual frequency totals per station.

    Parameters:
        stations_dict (dict): Dictionary of station names and DataFrames.
        percentile (float): Percentile threshold for extreme event classification.
        output_csv (str): Path to save exported CSV dataset.

    Returns:
        pd.DataFrame: Summary table with annual counts of extreme events per station.
    """
    station_summaries = []

    for station_name, df in stations_dict.items():
        if df is None or df.empty:
            continue

        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])

        if 'is_jbn' in df_copy.columns:
            df_copy = df_copy[df_copy['is_jbn'] == True]

        df_filtered = df_copy.dropna(subset=['wspd_ms'])
        if df_filtered.empty:
            continue

        threshold = np.percentile(df_filtered['wspd_ms'], percentile)
        df_extremes = df_filtered[df_filtered['wspd_ms'] >= threshold].copy()

        df_extremes['year'] = df_extremes['datetime'].dt.year
        counts = df_extremes.groupby('year').size().reset_index(name=station_name)
        station_summaries.append(counts)

    if not station_summaries:
        print("No extreme events found across datasets.")
        return pd.DataFrame()

    df_merged = station_summaries[0]
    for df_c in station_summaries[1:]:
        df_merged = pd.merge(df_merged, df_c, on='year', how='outer')

    station_cols = [col for col in df_merged.columns if col != 'year']
    df_merged[station_cols] = df_merged[station_cols].fillna(0).astype(int)
    df_merged = df_merged.sort_values('year').reset_index(drop=True)

    df_merged['total_events'] = df_merged[station_cols].sum(axis=1)
    df_summary = df_merged[df_merged['total_events'] > 0].copy()

    final_cols = ['year'] + station_cols + ['total_events']
    df_summary = df_summary[final_cols]

    df_summary.to_csv(output_csv, index=False)
    print(f"Annual dataset exported to '{output_csv}' containing {len(df_summary)} recorded years.")

    return df_summary


# ---------------------------------------------------------------------------
# Main Pipeline Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stations_dfs = []
    stations_dict = {}

    # 1. Load and pre-filter raw sounding data
    for name, filename, lat, lon in STATIONS_CONFIG:
        file_path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(file_path):
            raw_df = pd.read_csv(file_path, parse_dates=["datetime"])
            cleaned_df = filter_strict_data(raw_df)
            
            stations_dfs.append((name, cleaned_df, lat, lon))
            stations_dict[name] = cleaned_df
        else:
            print(f"Warning: File not found at '{file_path}'. Skipping station '{name}'.")

    if not stations_dfs:
        print("No valid dataset files found to process. Please check input paths.")
    else:
        # 2. Generate and save combined grid plot
        grid_output_path = os.path.join(OUTPUT_DIR, f"hist_series_sallj_p{TARGET_PERCENTILE}.png")
        plot_path = plot_complete_sallj_grid(
            stations_dfs,
            percentile=TARGET_PERCENTILE,
            min_persistence=MIN_PERSISTENCE_SOUNDINGS,
            gap_limit_days=365,
            out_path=grid_output_path,
            hist_color="#DD8452",
            ts_color="#4C72B0"
        )
        print(f"\nComplete analysis plot saved to: {plot_path}")

        # 3. Generate and save boxplot
        boxplot_output_path = os.path.join(OUTPUT_DIR, "sallj_boxplot.png")
        plot_station_boxplot(stations_dict, out_path=boxplot_output_path)
        print(f"Boxplot saved to: {boxplot_output_path}")

        # 4. Export monthly extremes summary
        monthly_csv_path = os.path.join(OUTPUT_DIR, "monthly_extremes_sallj.csv")
        df_monthly = generate_monthly_extremes_dataset(
            stations_dict=stations_dict,
            percentile=95,
            output_csv=monthly_csv_path
        )
        print("\nMonthly Extremes Dataset Sample:")
        print(df_monthly.head())

        # 5. Export annual extremes summary
        annual_csv_path = os.path.join(OUTPUT_DIR, "annual_extremes_sallj.csv")
        df_annual = generate_annual_extremes_dataset(
            stations_dict=stations_dict,
            percentile=95,
            output_csv=annual_csv_path
        )
        print("\nAnnual Extremes Dataset Sample:")
        print(df_annual.head(10))

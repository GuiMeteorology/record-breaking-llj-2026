#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: process_wyoming_data.py
Description: Performs preprocessing and standardization of downloaded University of
             Wyoming atmospheric sounding data into an IGRAv2-compatible format.

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

import glob
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & Metadata Mapping
# ---------------------------------------------------------------------------
STATION_NUMBER = "station_number"
INPUT_DIR = f"soundings_{STATION_NUMBER}"
OUTPUT_CSV = f"soundings_{STATION_NUMBER}_processed.csv"

# WMO station ID mapping to IGRAv2-formatted 11-character identifier.
# If a station is not present in this mapping, the raw WMO number is used.
STATION_ID_MAP = {
    "83554": "BRM00083554",  # Corumbá (Airport), MS - Brazil
}

# Fixed-width format specification for Wyoming sounding tables (11 columns x 7 chars)
COLSPECS = [
    (0, 7), (7, 14), (14, 21), (21, 28), (28, 35),
    (35, 42), (42, 49), (49, 56), (56, 63), (63, 70), (70, 77)
]

COLNAMES = [
    "PRES", "HGHT", "TEMP", "DWPT", "RELH",
    "MIXR", "DRCT", "SPED", "THTA", "THTE", "THTV"
]


def parse_wyoming_file(file_path: str) -> pd.DataFrame:
    """
    Parses a single raw Wyoming text file and extracts sounding data along with
    associated header metadata (ID, date, time, coordinates, total levels).

    Parameters:
        file_path (str): Path to the raw Wyoming text file.

    Returns:
        pd.DataFrame: Parsed sounding levels containing metadata and raw variables.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Locate and parse header metadata line
    header_line = next((line for line in lines if line.startswith("Observations for Station")), None)
    if header_line is None:
        raise ValueError(f"Header 'Observations for Station' not found in: {file_path}")

    header_match = re.search(
        r"Station\s+(\d+)\s+at\s+(\d{2})\s+UTC\s+(\d{2})\s+(\w{3})\s+(\d{4})", 
        header_line
    )
    if not header_match:
        raise ValueError(f"Unable to parse metadata header in {file_path}: {header_line.strip()}")
    
    station_num, hour_str, day_str, month_abbr, year_str = header_match.groups()

    year = int(year_str)
    month = datetime.strptime(month_abbr, "%b").month
    day = int(day_str)
    hour = int(hour_str)

    # Locate and parse geographical coordinates
    latlon_line = next((line for line in lines if line.strip().startswith("Latitude")), None)
    if latlon_line is None:
        raise ValueError(f"Geographical coordinates line not found in: {file_path}")
    
    lat = float(re.search(r"Latitude:\s*(-?\d+\.?\d*)", latlon_line).group(1))
    lon = float(re.search(r"Longitude:\s*(-?\d+\.?\d*)", latlon_line).group(1))

    # Extract fixed-width tabular data block (located after the second '---' divider)
    divider_indices = [idx for idx, line in enumerate(lines) if line.strip().startswith("---")]
    if len(divider_indices) < 2:
        raise ValueError(f"Separators '---' missing or incomplete in: {file_path}")
    
    data_start = divider_indices[1] + 1
    data_lines = [line for line in lines[data_start:] if line.strip() != ""]
    
    if not data_lines:
        raise ValueError(f"No sounding level records found in: {file_path}")

    data_block = "".join(data_lines)
    lvl_df = pd.read_fwf(
        io.StringIO(data_block), 
        colspecs=COLSPECS, 
        names=COLNAMES, 
        header=None
    )

    station_id = STATION_ID_MAP.get(station_num, station_num)

    # Assign metadata attributes to each level
    lvl_df["id"] = station_id
    lvl_df["year"] = year
    lvl_df["month"] = month
    lvl_df["day"] = day
    lvl_df["hour"] = hour
    lvl_df["lat"] = lat
    lvl_df["lon"] = lon
    lvl_df["numlev"] = len(lvl_df)

    return lvl_df


def build_igra_like_columns(lvl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Wyoming DataFrame fields into standardized IGRAv2-compatible 
    column names, units, and scaled integer values.

    Parameters:
        lvl_df (pd.DataFrame): Raw parsed Wyoming DataFrame.

    Returns:
        pd.DataFrame: Formatted DataFrame structured to mirror IGRAv2 dataset conventions.
    """
    df = lvl_df.copy()

    # Standardized floating-point variables (metric/SI units)
    df["press_hpa"] = df["PRES"]
    df["gph"] = df["HGHT"]                  # Geopotential height in meters
    df["temp_c"] = df["TEMP"]
    df["rh_pct"] = df["RELH"]
    df["dpdp_c"] = df["TEMP"] - df["DWPT"]  # Dewpoint depression (T - Td) in °C
    df["wspd_ms"] = df["SPED"]
    df["wdir"] = df["DRCT"]

    # Reconstructed integer variables with IGRAv2 scaling factors
    df["press"] = (df["press_hpa"] * 100).round()
    df["temp"] = (df["temp_c"] * 10).round()
    df["rh"] = (df["rh_pct"] * 10).round()
    df["dpdp"] = (df["dpdp_c"] * 10).round()
    df["wspd"] = (df["wspd_ms"] * 10).round()

    # Cast scaled fields to nullable Int64 type to handle missing values cleanly
    for col in ["press", "temp", "rh", "dpdp", "wspd"]:
        df[col] = df[col].astype("Int64")

    # Wyoming-exclusive thermodynamic quantities
    df["mixr"] = df["MIXR"]
    df["theta"] = df["THTA"]
    df["thte"] = df["THTE"]
    df["thtv"] = df["THTV"]

    # Generate consolidated UTC timestamp
    df["datetime"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
        errors="coerce"
    )

    # Standardized column ordering (IGRAv2 core fields + Wyoming extras)
    final_cols = [
        "id", "year", "month", "day", "hour", "numlev", "lat", "lon",
        "press", "gph", "temp", "rh", "dpdp", "wdir", "wspd",
        "press_hpa", "temp_c", "rh_pct", "dpdp_c", "wspd_ms",
        "datetime",
        "mixr", "theta", "thte", "thtv"
    ]
    
    return df[final_cols]


def process_wyoming_files(file_list: list, output_csv_path: str) -> pd.DataFrame:
    """
    Batch processes a collection of Wyoming text files, concatenates all valid
    profiles, and exports the resulting dataset to a consolidated CSV.

    Parameters:
        file_list (list): Paths to raw text files to process.
        output_csv_path (str): File path where the output CSV will be saved.

    Returns:
        pd.DataFrame: Combined DataFrame containing all processed soundings.
    """
    processed_dfs = []
    
    for path in file_list:
        try:
            lvl_df = parse_wyoming_file(path)
            formatted_df = build_igra_like_columns(lvl_df)
            processed_dfs.append(formatted_df)
        except Exception as err:
            print(f"Skipping {path} due to error: {err}")

    if not processed_dfs:
        raise ValueError("No valid sounding files were processed.")

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.to_csv(output_csv_path, index=True)
    
    return combined_df


if __name__ == "__main__":
    # Retrieve raw text files from the defined input directory
    input_files = sorted(glob.glob(f"{INPUT_DIR}/**/*.txt", recursive=True))

    if not input_files:
        print(f"No files found matching criteria in '{INPUT_DIR}'. Check configuration paths.")
    else:
        final_df = process_wyoming_files(input_files, OUTPUT_CSV)

        print(final_df.head())
        print(
            f"\nProcessing complete: {len(input_files)} file(s) processed, "
            f"{len(final_df)} total levels compiled into '{OUTPUT_CSV}'."
        )

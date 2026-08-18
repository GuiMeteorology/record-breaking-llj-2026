#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: montini_criteria.py
Description: Performs the classification of Low-Level Jet (LLJ) events according 
             to Montini et al. (2019) methodology (https://doi.org/10.1029/2018JD029634).

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd

# ==============================================================================
# GENERAL SETUP AND DIRECTORY CONFIGURATION
# ==============================================================================
CITY_NAME = "Corumba"

# Directory path definitions relative to repository root
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / f"{CITY_NAME}.csv"  # Accepts .csv or .parquet
OUTPUT_FILE = BASE_DIR / "data" / "processed" / f"{CITY_NAME}_LLJ_montini.csv"

START_DATE = "2007-01-01"
END_DATE = "2026-07-21"

# Target input dataset column mapping
COL_DATETIME = "datetime"
COL_PRESSURE = "press_hpa"
COL_DIR      = "wdir"
COL_SPEED    = "wspd_ms"

# METHODOLOGICAL PARAMETERS (Montini et al., 2019)
PRESSURE_LOW = 850        # hPa (Jet core level)
PRESSURE_HIGH = 700       # hPa (Upper level for vertical shear calculation)
PERCENTILE_THRESHOLD = 75  # 75th percentile for wind speed and scalar shear thresholds

# Sector defining northerly flow: NW-N-NE (292.5° to 45.0°)
DIR_MIN = 292.5
DIR_MAX = 45.0

# Seasonal mapping for climatological percentile calculations
SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON"
}

# ==============================================================================
# PROCESSING FUNCTIONS
# ==============================================================================

def apply_temporal_filter(df, start_date=START_DATE, end_date=END_DATE):
    """Filters the input DataFrame within the specified start and end date range."""
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df[(df[COL_DATETIME] >= pd.to_datetime(start_date)) & 
                (df[COL_DATETIME] <= pd.to_datetime(end_date))].copy()

    return df_out.sort_values(COL_DATETIME).reset_index(drop=True)


def load_data(filepath):
    """Loads CSV or Parquet file and extracts seasonal components."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
        
    filepath_str = str(filepath)
    if filepath_str.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
        
    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])
    df["month"] = df[COL_DATETIME].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    
    return df


def is_northerly_sector(deg):
    """Evaluates whether wind direction falls within the NW-N-NE sector (292.5° to 45.0°)."""
    return (deg >= DIR_MIN) | (deg <= DIR_MAX)


def prepare_wide_format(df):
    """Reshapes pressure level observations into dedicated columns for 850 and 700 hPa levels."""
    df_sub = df[df[COL_PRESSURE].isin([PRESSURE_LOW, PRESSURE_HIGH])].copy()
    
    wide = df_sub.pivot_table(
        index=[COL_DATETIME, "season"],
        columns=COL_PRESSURE,
        values=[COL_SPEED, COL_DIR]
    )
    
    wide.columns = [f"{col}_{int(p)}" for col, p in wide.columns]
    return wide.reset_index()


def compute_shear(wide):
    """Calculates scalar vertical wind shear (V_850 - V_700) per Montini et al. (2019)."""
    spd_low = wide[f"{COL_SPEED}_{PRESSURE_LOW}"]
    spd_high = wide[f"{COL_SPEED}_{PRESSURE_HIGH}"]
    
    wide["shear_ms"] = spd_low - spd_high
    return wide


def evaluate_soundings(wide, percentile=PERCENTILE_THRESHOLD):
    """
    Applies the LLJ classification criteria per sounding observation and logs justifications 
    for non-LLJ soundings.
    """
    spd_col = f"{COL_SPEED}_{PRESSURE_LOW}"
    dir_col = f"{COL_DIR}_{PRESSURE_LOW}"
    
    # Calculate seasonal 75th percentile thresholds
    thresh = wide.groupby("season").agg(
        speed_thresh=(spd_col, lambda s: np.nanpercentile(s, percentile)),
        shear_thresh=("shear_ms", lambda s: np.nanpercentile(s, percentile))
    ).reset_index()
    
    wide = wide.merge(thresh, on="season", how="left")
    
    # Evaluate the three Montini criteria
    cond_speed = wide[spd_col] >= wide["speed_thresh"]
    cond_shear = wide["shear_ms"] >= wide["shear_thresh"]
    cond_dir = is_northerly_sector(wide[dir_col])
    
    # Final classification flag
    wide["is_llj"] = cond_speed & cond_shear & cond_dir
    
    # Build textual explanations for soundings failing criteria
    def get_reason(row):
        if row["is_llj"]:
            return "LLJ criteria satisfied"
        
        reasons = []
        if not row["cond_speed"]:
            reasons.append(f"Low 850hPa wind speed ({row[spd_col]:.1f} < P75 {row['speed_thresh']:.1f} m/s)")
        if not row["cond_shear"]:
            reasons.append(f"Low vertical shear ({row['shear_ms']:.1f} < P75 {row['shear_thresh']:.1f} m/s)")
        if not row["cond_dir"]:
            reasons.append(f"Wind direction outside NW-N-NE sector ({row[dir_col]:.1f}°)")
            
        return " | ".join(reasons)

    # Assign boolean criteria checks to temporary evaluation columns
    wide["cond_speed"] = cond_speed
    wide["cond_shear"] = cond_shear
    wide["cond_dir"] = cond_dir
    
    wide["classification_reason"] = wide.apply(get_reason, axis=1)
    
    # Clean up temporary helper columns
    wide.drop(columns=["cond_speed", "cond_shear", "cond_dir"], inplace=True)
    
    return wide

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print(f"Reading and evaluating soundings for: {CITY_NAME}...")
    
    # 1. Load data
    df_raw = load_data(INPUT_FILE)
    df_temporal = apply_temporal_filter(df_raw)
    
    # 2. Reshape dataframe by pressure levels (850 and 700 hPa)
    wide_df = prepare_wide_format(df_temporal)
    
    # 3. Filter soundings with valid data across BOTH pressure levels
    req_cols = [
        f"{COL_SPEED}_{PRESSURE_LOW}", f"{COL_DIR}_{PRESSURE_LOW}",
        f"{COL_SPEED}_{PRESSURE_HIGH}", f"{COL_DIR}_{PRESSURE_HIGH}"
    ]
    wide_df = wide_df.dropna(subset=req_cols).copy()
    
    # 4. Compute shear and evaluate LLJ criteria
    wide_df = compute_shear(wide_df)
    df_result = evaluate_soundings(wide_df)
    
    # --------------------------------------------------------------------------
    # FINAL DATAFRAME FORMATTING
    # --------------------------------------------------------------------------
    cols_order = [
        COL_DATETIME, "season", "is_llj", "classification_reason",
        f"{COL_SPEED}_{PRESSURE_LOW}", "speed_thresh",
        "shear_ms", "shear_thresh",
        f"{COL_DIR}_{PRESSURE_LOW}"
    ]
    
    # Select and reorder output columns
    df_final = df_result[cols_order].sort_values(by=COL_DATETIME).copy()
    
    # Rename 850 hPa wind speed and direction back to standard column names
    df_final.rename(columns={
        f"{COL_SPEED}_{PRESSURE_LOW}": COL_SPEED,
        f"{COL_DIR}_{PRESSURE_LOW}": COL_DIR
    }, inplace=True)
    
    # Display final execution outcome
    print("\n--- Final Sounding Results Sample (First 15 rows) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_final.head(15))
    
    # Print statistical summary
    tot_soundings = len(df_final)
    tot_llj = df_final["is_llj"].sum()
    print(f"\nTotal valid soundings evaluated: {tot_soundings}")
    print(f"Total classified LLJ soundings: {tot_llj} ({tot_llj/tot_soundings*100:.1f}%)")

    # Ensure output directory exists before saving
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

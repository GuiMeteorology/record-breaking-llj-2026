#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: igra_data_process.py
Description: Performs preprocessing of IGRAv2 (Integrated Global Radiosonde Archive) data.

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Path definitions following project directory standards:
# BASE_DIR points to the repository root directory (one level up from 'scripts/')
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "BRM00083554-data.txt"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "BRM00083554_processed.csv"


def parse_igra2(path):
    """
    Parses an IGRAv2 station data file into a pandas DataFrame.
    Extracts header metadata (station ID, date, time, location) and links it
    to each vertical level observation.
    """
    headers = []
    data = []
    current_header = None

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            # Header lines start with '#' and contain metadata for the sounding launch
            if line.startswith("#"):
                current_header = {
                    "id": line[1:12].strip(),
                    "year": int(line[13:17]),
                    "month": int(line[18:20]),
                    "day": int(line[21:23]),
                    "hour": int(line[24:26]),
                    "reltime": line[27:31].strip(),
                    "numlev": int(line[32:36]),
                    "lat": int(line[55:62]) / 10000,
                    "lon": int(line[63:71]) / 10000,
                }
            # Data lines contain vertical level observations (pressure, height, temp, etc.)
            else:
                row = {
                    **current_header,
                    "lvltyp1": int(line[0:1]),
                    "lvltyp2": int(line[1:2]),
                    "etime": line[3:8].strip(),
                    "press": int(line[9:15]),
                    "gph": int(line[16:21]),
                    "temp": int(line[22:27]),
                    "rh": int(line[28:33]),
                    "dpdp": int(line[34:39]),
                    "wdir": int(line[40:45]),
                    "wspd": int(line[46:51]),
                }
                data.append(row)

    df = pd.DataFrame(data)

    # Replace missing/special indicator codes with NaN (-9999: missing, -8888/-7777: special flags)
    for col in ["press", "gph", "temp", "rh", "dpdp", "wdir", "wspd"]:
        df[col] = df[col].replace([-9999, -8888, -7777], np.nan)

    # Apply scale factors according to IGRAv2 format specification (igra2-data-format.txt)
    df["press_hpa"] = df["press"] / 100      # Pa -> hPa
    df["temp_c"]    = df["temp"] / 10        # Tenths of °C -> °C
    df["rh_pct"]    = df["rh"] / 10          # Scale adjustment for relative humidity
    df["dpdp_c"]    = df["dpdp"] / 10        # Tenths of °C -> °C (dewpoint depression)
    df["wspd_ms"]   = df["wspd"] / 10        # Tenths of m/s -> m/s

    # Create timestamp for each sounding observation
    df["datetime"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
        errors="coerce"
    )

    return df


if __name__ == "__main__":
    # Ensure destination directories exist prior to saving
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Execute parser and export processed dataset to CSV
    df = parse_igra2(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

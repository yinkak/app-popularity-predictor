#!/usr/bin/env python3
"""
combine_datasets.py

This script loads the cleaned FDroid and Google Play data, derives new features,
and combines them into a single CSV file for further analysis.

Folder structure assumed:
CMPT-353-PROJECT
├── data/
│   ├── cleaned/
│   │    ├── googleplay_cleaned.csv
│   │    └── fdroid_cleaned.csv
│   ├── uncleaned/
│   │    ├── googleplaystore.csv
│   │    └── fdroid.json
│   └── combined/
│        (the script creates this folder if it does not exist)
├── acquire_fdroid.py
├── acquire_googleplay.py
├── clean_fdroid.py
├── clean_googleplay.py
└── README.md

Usage:
    python3 combine_datasets.py
"""

import os
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_process_fdroid(fdroid_file: str) -> pd.DataFrame:
    """
    Loads the cleaned FDroid data CSV, processes the date, renames the 'categories'
    column to 'category' for consistency, and adds a 'platform' column.

    Args:
        fdroid_file (str): Path to the FDroid cleaned CSV file.
    Returns:
        pd.DataFrame: Processed FDroid DataFrame.
    """
    logging.info(f"Loading FDroid data from {fdroid_file}")
    df = pd.read_csv(fdroid_file)

    # For consistency, rename 'categories' to 'category' if necessary.
    if "categories" in df.columns:
        df.rename(columns={"categories": "category"}, inplace=True)

    # Convert last_updated_date to datetime (if not already)
    df["last_updated_date"] = pd.to_datetime(df["last_updated_date"], errors="coerce")

    # Add platform indicator
    df["platform"] = "FDroid"

    return df


def load_and_process_googleplay(gp_file: str) -> pd.DataFrame:
    """
    Loads the cleaned Google Play data CSV, ensures the 'category' field is present,
    converts the date, and adds a 'platform' column.

    Args:
        gp_file (str): Path to the Google Play cleaned CSV file.
    Returns:
        pd.DataFrame: Processed Google Play DataFrame.
    """
    logging.info(f"Loading Google Play data from {gp_file}")
    df = pd.read_csv(gp_file)

    # Ensure the column is named 'category'
    if "categories" in df.columns and "category" not in df.columns:
        df.rename(columns={"categories": "category"}, inplace=True)

    df["last_updated_date"] = pd.to_datetime(df["last_updated_date"], errors="coerce")

    # Add platform indicator
    df["platform"] = "Google Play"

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives additional features for the combined dataset.
    Currently creates 'app_age_days', which is the number of days since the app was last updated.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'last_updated_date' column.
    Returns:
        pd.DataFrame: DataFrame with derived features.
    """
    today = pd.Timestamp.today()
    logging.info(f"Deriving 'app_age_days' using today's date: {today.isoformat()}")
    df["app_age_days"] = (today - df["last_updated_date"]).dt.days
    return df


def combine_datasets(fdroid_df: pd.DataFrame, gp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the FDroid and Google Play DataFrames into a single DataFrame.

    Args:
        fdroid_df (pd.DataFrame): Processed FDroid DataFrame.
        gp_df (pd.DataFrame): Processed Google Play DataFrame.
    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    combined_df = pd.concat([fdroid_df, gp_df], ignore_index=True, sort=False)
    logging.info(f"Combined dataset contains {len(combined_df)} records.")
    return combined_df


def main():
    # Define file paths using the updated folder structure
    fdroid_file = "data/cleaned/fdroid_cleaned.csv"
    gp_file = "data/cleaned/googleplay_cleaned.csv"
    output_file = "data/combined/combined_apps.csv"

    # Ensure the combined data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load and process each dataset
    fdroid_df = load_and_process_fdroid(fdroid_file)
    gp_df = load_and_process_googleplay(gp_file)

    # Derive additional features
    fdroid_df = derive_features(fdroid_df)
    gp_df = derive_features(gp_df)

    # Combine the datasets
    combined_df = combine_datasets(fdroid_df, gp_df)

    # Save the combined dataset
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Combined dataset saved to {output_file}")


if __name__ == "__main__":
    main()

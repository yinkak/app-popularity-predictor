#!/usr/bin/env python3
"""
clean_googleplay.py

Cleans the raw Google Play Store Applications CSV data downloaded from Kaggle.
The script converts key columns to proper data types, standardizes field names,
and handles ambiguous values by marking them as missing. Rows missing critical
fields are dropped. The cleaned data is saved to a new CSV file for further analysis.

Usage:
    python3 clean_googleplay.py
"""

import os
import re
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_size(size_str: str) -> float:
    """
    Cleans the 'Size' field and converts it to a numeric value representing MB.
    Handles sizes formatted as "19M", "14M", "8.7M", or "100K". If the format is ambiguous
    (e.g., "1,000+") or not recognized, returns None.

    Args:
        size_str (str): A string representing the app size.

    Returns:
        float: The size in MB, or None if the value is not valid.
    """
    if pd.isna(size_str) or size_str.lower().strip() == "varies with device":
        return None
    # Remove commas and plus signs
    cleaned = size_str.replace(",", "").replace("+", "").strip()
    try:
        if cleaned.lower().endswith("m"):
            return float(cleaned[:-1])
        elif cleaned.lower().endswith("k"):
            return float(cleaned[:-1]) / 1024.0
        else:
            # If there's no trailing letter, attempt conversion only if the string is numeric
            if re.fullmatch(r"\d+(\.\d+)?", cleaned):
                return float(cleaned)
            else:
                # Ambiguous format encountered
                logging.warning(
                    f"Ambiguous Size format '{size_str}'; marking as missing."
                )
                return None
    except Exception as e:
        logging.warning(f"Error converting Size '{size_str}': {e}")
        return None


def clean_installs(installs_str: str) -> int:
    """
    Cleans the 'Installs' field by removing commas and plus signs,
    then converting it to an integer. If the field contains "Free" or
    non-numeric values, returns None.

    Args:
        installs_str (str): A string such as "10,000+".

    Returns:
        int: The number of installs, or None if conversion fails.
    """
    if pd.isna(installs_str):
        return None
    if installs_str.strip().lower() == "free":
        return None
    cleaned = installs_str.replace(",", "").replace("+", "").strip()
    try:
        return int(cleaned)
    except Exception as e:
        logging.warning(f"Error converting Installs '{installs_str}': {e}")
        return None

def assign_price_tier(price):
    """
    Assigns an ordinal price tier (1–5) based on app price.

    Args:
        price (float or int): The cleaned numeric price.

    Returns:
        int: Ordinal price tier. 1 = free, 5 = most expensive.
    """
    if pd.isna(price):
        return 1  # Default unknown/missing as free
    elif price == 0:
        return 1
    elif price <= 2:
        return 2
    elif price <= 10:
        return 3
    elif price <= 30:
        return 4
    else:
        return 5


def clean_googleplay_data(
    input_file: str = "data/uncleaned/googleplaystore.csv",
    output_file: str = "data/cleaned/googleplay_cleaned.csv",
) -> None:
    """
    Reads the raw Google Play Store Applications CSV data, cleans and processes it,
    and saves the cleaned data to a new CSV file.

    Args:
        input_file (str): Path to the raw CSV file.
        output_file (str): Path to save the cleaned CSV file.
    """
    logging.info(f"Reading raw data from {input_file}")
    # Read CSV; assume the first column is an index column
    df = pd.read_csv(input_file, index_col=0)

    # Normalize column names: trim whitespace and convert to lower-case
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Standardize column names (if necessary)
    df.rename(
        columns={
            "app": "app_name",
            "category": "category",
            "rating": "rating",
            "reviews": "reviews",
            "size": "size",
            "installs": "installs",
            "type": "type",
            "price": "price",
            "content rating": "content_rating",
            "genres": "genres",
            "last updated": "last_updated",
            "current ver": "current_ver",
            "android ver": "android_ver",
        },
        inplace=True,
    )

    logging.info("Cleaning numerical fields...")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce", downcast="integer")
    df["size_mb"] = df["size"].apply(clean_size)
    df["installs_clean"] = df["installs"].apply(clean_installs)

    # Clean price field: remove currency symbols (using raw string literal for \$)
    df["price_clean"] = df["price"].replace({r"\$": "", ",": ""}, regex=True)
    df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")

    # Derive price tier from price_clean
    df["price_tier"] = df["price_clean"].apply(assign_price_tier)


    logging.info("Converting 'last_updated' column to datetime...")
    try:
        df["last_updated_date"] = pd.to_datetime(
            df["last_updated"], format="%d-%b-%y", errors="coerce"
        )
    except Exception as e:
        logging.error(f"Error parsing 'last_updated' dates: {e}")

    # Compute days since last update
    df["update_freq_days"] = (pd.Timestamp("today") - df["last_updated_date"]).dt.days

    # Assume Google Play apps do not report anti-features; default to 0
    df["anti_feature_score"] = 0

    initial_len = len(df)
    # Drop rows with missing critical data: app_name, rating, installs_clean
    df.dropna(subset=["app_name", "rating", "installs_clean"], inplace=True)
    logging.info(f"Dropped {initial_len - len(df)} rows missing critical fields.")

    # Create a cleaned dataframe with the needed columns.
    df_clean = df[
        [
            "app_name",
            "category",
            "rating",
            "reviews",
            "size_mb",
            "installs_clean",
            "type",
            "price_clean",
            "price_tier",
            "content_rating",
            "genres",
            "last_updated_date",
            "current_ver",
            "android_ver",
            "update_freq_days",
            "anti_feature_score",
        ]
    ].copy()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    logging.info(f"Cleaned data saved to {output_file}")


def main():
    input_file = "data/uncleaned/googleplaystore.csv"
    output_file = "data/cleaned/googleplay_cleaned.csv"
    clean_googleplay_data(input_file, output_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
clean_fdroid.py

Enhanced script to clean FDroid JSON data with robust error handling,
comprehensive field coverage, and localization fallbacks.
The cleaned data is saved as a CSV file for further analysis and comparison.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_fdroid_data(input_file: str) -> Dict[str, Any]:
    """
    Loads FDroid JSON data from a file and validates that it contains the expected key 'apps'.

    Args:
        input_file (str): Path to the raw FDroid JSON file.
    Returns:
        Dict[str, Any]: The parsed JSON data.
    Raises:
        KeyError: If the key 'apps' is missing.
        Exception: For any I/O or JSON decoding errors.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "apps" not in data:
            raise KeyError("JSON missing 'apps' key.")
        return data
    except Exception as e:
        logging.error(f"Failed to load FDroid data from {input_file}: {e}")
        raise


def get_localized_field(
    localized: Dict[str, Any],
    field: str,
    lang_priority: List[str] = ["en-US", "en", "de"],
) -> Optional[str]:
    """
    Extracts a localized field from the 'localized' dictionary using a priority list of language codes.

    Args:
        localized (Dict[str, Any]): Dictionary containing localized data.
        field (str): Field to extract, e.g. "name" or "summary".
        lang_priority (List[str]): List of preferred language codes in order.
    Returns:
        Optional[str]: The localized field value if found; None otherwise.
    """
    for lang in lang_priority:
        if lang in localized and field in localized[lang]:
            return localized[lang][field]
    return None


def process_app(app: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes a single app entry from the FDroid JSON data.
    Converts timestamps, extracts nested localized fields, and joins list fields.

    Args:
        app (Dict[str, Any]): A dictionary representing one app.
    Returns:
        Optional[Dict[str, Any]]: Processed app data as a dictionary, or None if mandatory fields are missing.
    """
    try:
        package_name = app.get("packageName")
        if not package_name:
            logging.warning("Skipping app with missing packageName.")
            return None

        # Extract basic metadata
        categories = ", ".join(app.get("categories", []))
        license_str = app.get("license", "Unknown")
        source_code = app.get("sourceCode")
        author = app.get("authorName", app.get("authorEmail"))
        suggested_version = app.get("suggestedVersionName")
        anti_features = ", ".join(app.get("antiFeatures", []))

        # Convert timestamps (if available), assuming they are in milliseconds
        added = app.get("added")
        last_updated = app.get("lastUpdated")
        added_date = datetime.fromtimestamp(added / 1000).isoformat() if added else None
        last_updated_date = (
            datetime.fromtimestamp(last_updated / 1000).isoformat()
            if last_updated
            else None
        )

        # Process localized fields with fallback
        localized = app.get("localized", {})
        app_name = get_localized_field(localized, "name")
        summary = get_localized_field(localized, "summary")
        description = get_localized_field(localized, "description")

        processed_app = {
            "package_name": package_name,
            "categories": categories,
            "license": license_str,
            "source_code": source_code,
            "author": author,
            "suggested_version": suggested_version,
            "anti_features": anti_features,
            "added_date": added_date,
            "last_updated_date": last_updated_date,
            "app_name": app_name,
            "summary": summary,
            "description": description,
        }
        return processed_app
    except Exception as e:
        logging.error(f"Error processing app {app.get('packageName')}: {e}")
        return None


def main(
    input_file: str = "data/fdroid.json",
    output_file: str = "data/cleaned/fdroid_cleaned.csv",
) -> None:
    """
    Main cleaning pipeline:
    Loads raw FDroid JSON data, processes each app entry, and saves a cleaned CSV.

    Args:
        input_file (str, optional): Path to the raw FDroid JSON file.
        output_file (str, optional): Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info("Loading FDroid data...")
    data = load_fdroid_data(input_file)
    apps = data["apps"]
    logging.info(f"Found {len(apps)} apps in the dataset.")

    # Process each app using list comprehension and filter out None values
    processed_apps = [process_app(app) for app in apps]
    processed_apps = [app for app in processed_apps if app is not None]

    # Create DataFrame and remove duplicates based on package_name
    df = pd.DataFrame(processed_apps)
    initial_count = len(df)
    df.drop_duplicates(subset="package_name", inplace=True)
    logging.info(
        f"Dropped {initial_count - len(df)} duplicate entries based on package_name."
    )

    # Save cleaned DataFrame to CSV
    df.to_csv(output_file, index=False)
    logging.info(f"Saved cleaned data to {output_file} ({len(df)} apps).")


if __name__ == "__main__":
    main()

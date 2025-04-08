#!/usr/bin/env python3
"""
acquire_googleplay.py

This script downloads the Google Play Store Applications dataset from Kaggle
using the Kaggle API. It downloads and unzips the dataset into a local directory
for further analysis.

Requirements:
- Kaggle API installed (pip install kaggle)
- Kaggle credentials configured in ~/.kaggle/kaggle.json

Usage:
    python3 acquire_googleplay.py
"""

import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_googleplay_dataset(
    dataset: str = "bhavikjikadara/google-play-store-applications",
    output_path: str = "data/googleplay",
) -> None:
    """
    Downloads and unzips the Google Play Store Applications dataset from Kaggle.

    Args:
        dataset (str): The Kaggle dataset identifier.
        output_path (str): Local directory to save and extract the dataset.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Output directory ensured: {output_path}")

    # Instantiate and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    logging.info("Authenticated with Kaggle API.")

    try:
        logging.info(f"Downloading dataset '{dataset}' to '{output_path}'...")
        api.dataset_download_files(dataset, path=output_path, unzip=True)
        logging.info("Dataset downloaded and extracted successfully.")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise


def main() -> None:
    dataset = "bhavikjikadara/google-play-store-applications"
    output_path = "data/googleplay"
    download_googleplay_dataset(dataset, output_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
acquire_fdroid.py

Fetches and processes F-Droid app metadata using the FDroid repository API.
Utilizes a robust requests session with retry logic and logging.
Saves structured data to a JSON file for further analysis.
"""

import os
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_session_with_retries(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)
):
    """
    Creates a requests session with retry logic.

    Args:
        retries (int): Maximum number of retry attempts.
        backoff_factor (float): Factor to compute sleep time between retries.
        status_forcelist (tuple): HTTP status codes that trigger a retry.

    Returns:
        requests.Session: A session object with the configured HTTPAdapter.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=[
            "GET"
        ],  # for newer versions, use allowed_methods instead of method_whitelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def acquire_fdroid_data(
    url="https://f-droid.org/repo/index-v1.json",
    output_file="data/uncleaned/fdroid.json",
    timeout=10,
):
    """
    Fetches F-Droid app metadata using the repository API, validates its structure,
    and saves the data as formatted JSON.

    Args:
        url (str): URL to the FDroid repository index in JSON format.
        output_file (str): Local file path to store the output JSON.
        timeout (int): Timeout (in seconds) for the HTTP request.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory ensured: {output_dir}")

    session = create_session_with_retries()

    try:
        logging.info(f"Fetching FDroid data from: {url}")
        response = session.get(url, timeout=timeout)
        logging.info(f"HTTP Status Code: {response.status_code}")
        response.raise_for_status()

        data = response.json()

        # Validate the JSON structure (check for expected keys)
        # Depending on the endpoint, the key might be "apps", "packages" or similar.
        if not isinstance(data, dict) or not any(
            key in data for key in ("apps", "packages")
        ):
            raise ValueError(
                "Invalid JSON structure: Expected key 'apps' or 'packages' not found."
            )

        # Save JSON data to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Data successfully acquired and saved to '{output_file}'.")

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        raise
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
        raise
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred: {timeout_err}")
        raise
    except json.JSONDecodeError as json_err:
        logging.error(f"Error decoding JSON: {json_err}")
        raise
    except ValueError as val_err:
        logging.error(f"Data validation error: {val_err}")
        raise
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
        raise


if __name__ == "__main__":
    acquire_fdroid_data()

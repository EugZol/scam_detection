import zipfile
from pathlib import Path

import requests


def download_data(
    url: str = "https://example.com/Phishing_Email.csv.zip", data_dir: str = "data"
) -> None:
    """
    Download the phishing email dataset from the given URL and unzip it.

    Args:
        url: URL to download the zip file from
        data_dir: Directory to save the data
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    zip_path = data_path / "Phishing_Email.csv.zip"
    csv_path = data_path / "Phishing_Email.csv"

    if csv_path.exists():
        print(f"Data already exists at {csv_path}")
        return

    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(response.content)

    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    print(f"Data downloaded and unzipped to {csv_path}")

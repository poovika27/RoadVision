"""Download the Kaggle dataset. Needs a Kaggle API token (never hard-code it).

Setup: kaggle.com -> Settings -> API -> Create New Token, then either
  - save kaggle.json to C:\\Users\\<you>\\.kaggle\\kaggle.json, or
  - set env var KAGGLE_API_TOKEN (new-style token).
"""
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = sys.argv[1] if len(sys.argv) > 1 else "ziya07/speed-bump-dataset"
OUT = Path(__file__).parent / "data" / "raw" / DATASET.split("/")[-1]

api = KaggleApi()
api.authenticate()
OUT.mkdir(parents=True, exist_ok=True)
api.dataset_download_files(DATASET, path=str(OUT), unzip=True)
print("Downloaded to", OUT)

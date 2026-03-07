"""
Seren ML Pipeline — WESAD Dataset Downloader
==============================================
Downloads the WESAD dataset from the UCI Machine Learning Repository.

WESAD: Wearable Stress and Affect Detection
- 15 subjects (S2-S17, excluding S12)
- Empatica E4 wrist sensor + RespiBAN chest sensor
- Labels: baseline, stress (TSST), amusement, meditation

Source: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx
"""

import os
import sys
import zipfile
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# WESAD is hosted on the university's cloud storage
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
DATA_DIR = Path(__file__).parent.parent / "data"
WESAD_DIR = DATA_DIR / "wesad"
WESAD_ZIP = DATA_DIR / "WESAD.zip"


def download_wesad():
    """Download and extract the WESAD dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    subject_dirs = list(WESAD_DIR.glob("S*")) if WESAD_DIR.exists() else []
    if len(subject_dirs) >= 14:
        logger.info(f"WESAD already downloaded: {len(subject_dirs)} subjects found in {WESAD_DIR}")
        return True

    # Download
    if not WESAD_ZIP.exists():
        logger.info(f"Downloading WESAD dataset (~2.2 GB)...")
        logger.info(f"URL: {WESAD_URL}")
        logger.info(f"Destination: {WESAD_ZIP}")

        try:
            import urllib.request

            def _progress(block_count, block_size, total_size):
                downloaded = block_count * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
                    sys.stdout.flush()

            urllib.request.urlretrieve(WESAD_URL, str(WESAD_ZIP), reporthook=_progress)
            print()  # newline after progress
            logger.info(f"Download complete: {WESAD_ZIP.stat().st_size / (1024**3):.2f} GB")

        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("You can manually download WESAD from:")
            logger.info("  https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx")
            logger.info(f"  Place the ZIP file at: {WESAD_ZIP}")
            if WESAD_ZIP.exists():
                WESAD_ZIP.unlink()
            return False

    # Extract
    logger.info(f"Extracting WESAD dataset...")
    try:
        with zipfile.ZipFile(str(WESAD_ZIP), 'r') as zf:
            zf.extractall(str(DATA_DIR))

        # The ZIP extracts to DATA_DIR/WESAD/ — rename to DATA_DIR/wesad/
        extracted_dir = DATA_DIR / "WESAD"
        if extracted_dir.exists() and not WESAD_DIR.exists():
            extracted_dir.rename(WESAD_DIR)
        elif extracted_dir.exists() and WESAD_DIR.exists():
            # Merge
            for item in extracted_dir.iterdir():
                target = WESAD_DIR / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(str(extracted_dir), ignore_errors=True)

        # Verify
        subject_dirs = list(WESAD_DIR.glob("S*"))
        logger.info(f"Extracted {len(subject_dirs)} subjects: {[d.name for d in sorted(subject_dirs)]}")

        # Clean up ZIP to save space
        if WESAD_ZIP.exists() and len(subject_dirs) >= 10:
            logger.info("Removing ZIP file to save disk space...")
            WESAD_ZIP.unlink()

        return len(subject_dirs) >= 10

    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid ZIP. Deleting and retrying may help.")
        if WESAD_ZIP.exists():
            WESAD_ZIP.unlink()
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    success = download_wesad()
    if success:
        print("\nWESAD dataset ready!")
        # List subjects
        for d in sorted(WESAD_DIR.glob("S*")):
            pkl = d / f"{d.name}.pkl"
            size = pkl.stat().st_size / (1024**2) if pkl.exists() else 0
            print(f"  {d.name}: {size:.0f} MB")
    else:
        print("\nFailed to download WESAD dataset.")
        sys.exit(1)

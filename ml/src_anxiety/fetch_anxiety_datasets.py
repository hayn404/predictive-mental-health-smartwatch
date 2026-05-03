import argparse
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_globem() -> None:
    """
    Fetch GLOBEM from PhysioNet using wfdb.
    Requires a configured PhysioNet credentialed account.
    """
    target = DATA_DIR / "globem"
    ensure_dir(target / "raw")
    print("\nGLOBEM uses PhysioNet credentialed access.")
    print("Dataset page: https://physionet.org/content/globem/")
    print("After access approval, download into:", target / "raw")
    print("Then prepare:")
    print("-", target / "sensor_features.csv")
    print("-", target / "anxiety_labels.csv")


def print_studentlife_instructions() -> None:
    target = DATA_DIR / "studentlife"
    ensure_dir(target)
    print("\nStudentLife requires manual download from Dartmouth.")
    print("Official page: https://studentlife.cs.dartmouth.edu/")
    print("Place extracted files under:", target / "raw")
    print("Then create:")
    print("-", target / "sensor_features.csv")
    print("-", target / "anxiety_labels.csv")


def print_kemophone_instructions() -> None:
    target = DATA_DIR / "k-emophone"
    ensure_dir(target)
    print("\nK-EmoPhone is access-controlled on Zenodo.")
    print("Request access at: https://zenodo.org/records/7606611")
    print("Use same email for Zenodo request and EULA.")
    print("Place extracted files under:", target / "raw")
    print("Then create:")
    print("-", target / "sensor_features.csv")
    print("-", target / "anxiety_labels.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch anxiety datasets for anxiety_model_2.")
    parser.add_argument(
        "--dataset",
        choices=["all", "globem", "studentlife", "kemophone"],
        default="all",
        help="Dataset to fetch or prepare",
    )
    args = parser.parse_args()

    if args.dataset in ("all", "globem"):
        fetch_globem()
    if args.dataset in ("all", "studentlife"):
        print_studentlife_instructions()
    if args.dataset in ("all", "kemophone"):
        print_kemophone_instructions()

    print("\nDone. No synthetic data generated.")


if __name__ == "__main__":
    main()

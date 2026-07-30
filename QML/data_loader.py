import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def _kaggle_credentials_available():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    return (
        (kaggle_dir / "access_token").exists()
        or (kaggle_dir / "access_token.txt").exists()
        or (kaggle_dir / "kaggle.json").exists()
    )


def _download_from_kaggle():
    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    matches = list(dataset_dir.rglob("creditcard.csv"))
    if not matches:
        raise FileNotFoundError(f"creditcard.csv not found under downloaded dataset at {dataset_dir}")
    return matches[0]


def load_creditcard_data(csv_path=None):
    if csv_path:
        path = Path(csv_path)
    elif CSV_PATH.exists():
        path = CSV_PATH
    elif _kaggle_credentials_available():
        print(f"Downloading {KAGGLE_DATASET} from Kaggle via kagglehub...")
        path = _download_from_kaggle()
        print("Path to dataset files:", path)
    else:
        path = CSV_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No dataset found at {path} and Kaggle credentials are not configured.\n"
            "Either place creditcard.csv there, or set up Kaggle API access, e.g.:\n"
            "  kaggle auth login\n"
            "or generate a token at https://www.kaggle.com/settings/api and either:\n"
            "  export KAGGLE_API_TOKEN=<token>\n"
            "  # or save it to ~/.kaggle/access_token\n"
            "  # or (legacy) save kaggle.json to ~/.kaggle/kaggle.json"
        )

    return pd.read_csv(path)


def summarize(df):
    print("Shape:", df.shape)

    counts = df["Class"].value_counts().sort_index()
    total = len(df)
    print("Class balance:")
    for cls, count in counts.items():
        label = "Fraud" if cls == 1 else "Legit"
        print(f"  {label} (Class={cls}): {count} ({count / total:.4%})")

    missing = int(df.isnull().sum().sum())
    if missing == 0:
        print("No missing values.")
    else:
        print(f"WARNING: {missing} missing values found:")
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0])


if __name__ == "__main__":
    data = load_creditcard_data()
    summarize(data)

import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FLIGHTS_PATH = os.path.join(RAW_DIR, "flights.csv")
AIRLINES_PATH = os.path.join(RAW_DIR, "airlines.csv")
AIRPORTS_PATH = os.path.join(RAW_DIR, "airports.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "flights_clean.csv")

SAMPLE_N = 250_000
RANDOM_STATE = 42
DELAY_THRESHOLD = 15


def load_raw_flights() -> pd.DataFrame:
    """Sample 100K rows from flights.csv."""
    df = pd.read_csv(FLIGHTS_PATH, low_memory=False)
    return df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)


def merge_reference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge airline names and airport details."""
    airlines = pd.read_csv(AIRLINES_PATH)
    airports = pd.read_csv(AIRPORTS_PATH)

    # Rename to avoid column collisions
    airlines = airlines.rename(columns={"AIRLINE": "AIRLINE_NAME", "IATA_CODE": "AIRLINE"})
    airports = airports.rename(columns={
        "IATA_CODE": "ORIGIN_AIRPORT",
        "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "CITY": "ORIGIN_CITY",
        "STATE": "ORIGIN_STATE",
        "LATITUDE": "ORIGIN_LAT",
        "LONGITUDE": "ORIGIN_LON",
    })

    df = df.merge(airlines, on="AIRLINE", how="left")
    df = df.merge(airports, on="ORIGIN_AIRPORT", how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled/diverted flights and rows with null ARRIVAL_DELAY."""
    df = df[df["CANCELLED"] == 0]
    df = df[df["DIVERTED"] == 0]
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 if ARRIVAL_DELAY >= 15, else 0."""
    df = df.copy()
    df["delayed"] = (df["ARRIVAL_DELAY"] >= DELAY_THRESHOLD).astype(int)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nNull counts (non-zero only):")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    print(null_counts.to_string() if not null_counts.empty else "  None")

    print("\nClass balance:")
    counts = df["delayed"].value_counts()
    total = len(df)
    on_time = counts.get(0, 0)
    delayed = counts.get(1, 0)
    print(f"  On-time (0): {on_time:,}  ({100 * on_time / total:.1f}%)")
    print(f"  Delayed (1): {delayed:,}  ({100 * delayed / total:.1f}%)")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading flights sample...")
    df = load_raw_flights()

    print("Merging reference data...")
    df = merge_reference_data(df)

    print("Cleaning...")
    df = clean(df)

    print("Adding target column...")
    df = add_target(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    print_summary(df)


if __name__ == "__main__":
    main()

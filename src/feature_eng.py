import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flights_clean.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "delayed"

# US federal holidays (month, day) + eve/day-after approximations
_HOLIDAY_DATES = {
    # New Year's
    (1, 1), (12, 31), (1, 2),
    # MLK Day — 3rd Monday Jan, approximate as Jan 15–21 window; use fixed anchor
    (1, 19),
    # Presidents Day — 3rd Monday Feb
    (2, 16),
    # Memorial Day — last Monday May
    (5, 25),
    # Independence Day
    (7, 3), (7, 4), (7, 5),
    # Labor Day — 1st Monday Sep
    (9, 7),
    # Thanksgiving — 4th Thursday Nov (approximate)
    (11, 25), (11, 26), (11, 27),
    # Christmas
    (12, 24), (12, 25), (12, 26),
}

HOLIDAY_SET = frozenset(_HOLIDAY_DATES)


def extract_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Hour & time-of-day
    out["hour"] = (df["DEPARTURE_TIME"] // 100).astype("Int64").astype(float)

    def time_of_day(h):
        if pd.isna(h):
            return "unknown"
        h = int(h)
        if 5 <= h <= 11:
            return "morning"
        elif 12 <= h <= 17:
            return "afternoon"
        elif 18 <= h <= 21:
            return "evening"
        else:
            return "night"

    out["time_of_day"] = out["hour"].map(time_of_day)

    # Day / month
    out["day_of_week"] = df["DAY_OF_WEEK"].astype(float)
    out["is_weekend"] = (df["DAY_OF_WEEK"].isin([6, 7])).astype(int)
    out["month"] = df["MONTH"].astype(float)

    # Holiday flag
    out["is_holiday"] = list(
        zip(df["MONTH"].astype(int), df["DAY"].astype(int))
    )
    out["is_holiday"] = out["is_holiday"].apply(lambda x: int(x in HOLIDAY_SET))

    # Distance bin
    def dist_bin(d):
        if pd.isna(d):
            return "unknown"
        if d < 500:
            return "short"
        elif d <= 1500:
            return "medium"
        else:
            return "long"

    out["distance_bin"] = df["DISTANCE"].map(dist_bin)

    # Keep raw keys for target-encoding lookups
    out["_airline"] = df["AIRLINE"].values
    out["_origin"] = df["ORIGIN_AIRPORT"].values
    out["_route"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)

    return out


def compute_delay_rates(features: pd.DataFrame, target: pd.Series) -> tuple[dict, dict, dict, float]:
    """Compute per-group delay rates from training data."""
    tmp = features[["_airline", "_origin", "_route"]].copy()
    tmp["y"] = target.values

    overall = tmp["y"].mean()

    carrier_rate = tmp.groupby("_airline")["y"].mean().to_dict()
    origin_rate = tmp.groupby("_origin")["y"].mean().to_dict()
    route_rate = tmp.groupby("_route")["y"].mean().to_dict()

    return carrier_rate, origin_rate, route_rate, overall


def apply_delay_rates(features: pd.DataFrame, carrier_rate, origin_rate, route_rate, overall) -> pd.DataFrame:
    features = features.copy()
    features["carrier_delay_rate"] = features["_airline"].map(carrier_rate).fillna(overall)
    features["origin_delay_rate"] = features["_origin"].map(origin_rate).fillna(overall)
    features["route_delay_rate"] = features["_route"].map(route_rate).fillna(overall)
    return features


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    tod_order = ["morning", "afternoon", "evening", "night", "unknown"]
    df["time_of_day"] = pd.Categorical(df["time_of_day"], categories=tod_order).codes.astype(float)

    dist_order = ["short", "medium", "long", "unknown"]
    df["distance_bin"] = pd.Categorical(df["distance_bin"], categories=dist_order).codes.astype(float)

    return df


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Extracting base features...")
    features = extract_base_features(df)
    target = df[TARGET]

    # Drop raw lookup keys before saving — used only internally
    feature_cols_no_keys = [c for c in features.columns if not c.startswith("_")]

    print("Splitting train/test (80/20, stratified)...")
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE)

    train_feat = features.iloc[train_idx].copy()
    test_feat = features.iloc[test_idx].copy()
    y_train = target.iloc[train_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)

    print("Computing delay rates from training data only...")
    carrier_rate, origin_rate, route_rate, overall = compute_delay_rates(train_feat, y_train)

    train_feat = apply_delay_rates(train_feat, carrier_rate, origin_rate, route_rate, overall)
    test_feat = apply_delay_rates(test_feat, carrier_rate, origin_rate, route_rate, overall)

    # Drop internal key columns
    train_feat = train_feat[feature_cols_no_keys + ["carrier_delay_rate", "origin_delay_rate", "route_delay_rate"]]
    test_feat = test_feat[feature_cols_no_keys + ["carrier_delay_rate", "origin_delay_rate", "route_delay_rate"]]

    print("Encoding categoricals...")
    train_feat = encode_categoricals(train_feat)
    test_feat = encode_categoricals(test_feat)

    train_feat = train_feat.reset_index(drop=True)
    test_feat = test_feat.reset_index(drop=True)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_feat.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    test_feat.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # Summary
    print(f"\nFeatures ({len(train_feat.columns)}):")
    print(" ", list(train_feat.columns))

    print(f"\nTrain shape : {train_feat.shape}")
    print(f"Test  shape : {test_feat.shape}")

    tr_pos = y_train.mean() * 100
    te_pos = y_test.mean() * 100
    print(f"\nClass balance — Train: {tr_pos:.1f}% delayed / {100-tr_pos:.1f}% on-time")
    print(f"Class balance — Test : {te_pos:.1f}% delayed / {100-te_pos:.1f}% on-time")
    print("\nSaved X_train, X_test, y_train, y_test to data/processed/")


if __name__ == "__main__":
    main()

"""
Flight booking optimizer — prescriptive recommendations using the tuned model.
All public functions lazy-load the model and training data on first call.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "models", "best_model_tuned.pkl")
DATA_PATH  = os.path.join(ROOT, "data", "processed", "flights_clean.csv")

RANDOM_STATE = 42

HOLIDAY_SET = frozenset([
    (1,1),(12,31),(1,2),(1,19),(2,16),(5,25),
    (7,3),(7,4),(7,5),(9,7),(11,25),(11,26),(11,27),
    (12,24),(12,25),(12,26),
])

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_model        = None
_train        = None
_overall      = None
_carrier_rate = None
_origin_rate  = None
_route_rate   = None
_dist_lookup  = None
_airline_name = None


def _load():
    global _model, _train, _overall, _carrier_rate, _origin_rate
    global _route_rate, _dist_lookup, _airline_name

    if _model is not None:
        return

    _model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    _train, _ = train_test_split(df, test_size=0.20, stratify=df["delayed"],
                                 random_state=RANDOM_STATE)

    _overall      = _train["delayed"].mean()
    _carrier_rate = _train.groupby("AIRLINE")["delayed"].mean().to_dict()
    _origin_rate  = _train.groupby("ORIGIN_AIRPORT")["delayed"].mean().to_dict()
    route_key     = (_train["ORIGIN_AIRPORT"].astype(str) + "_"
                     + _train["DESTINATION_AIRPORT"].astype(str))
    _route_rate   = _train.assign(_route=route_key).groupby("_route")["delayed"].mean().to_dict()
    _dist_lookup  = (
        _train.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["DISTANCE"]
        .median().to_dict()
    )
    _airline_name = (
        _train[["AIRLINE", "AIRLINE_NAME"]]
        .dropna().drop_duplicates()
        .set_index("AIRLINE")["AIRLINE_NAME"]
        .to_dict()
    )


# ── Shared helpers ────────────────────────────────────────────────────────────
def _fmt_hour(h):
    if h == 0:  return "12am"
    if h < 12:  return f"{h}am"
    if h == 12: return "12pm"
    return f"{h - 12}pm"


def _route_distance(origin, destination):
    dist = _dist_lookup.get((origin, destination), np.nan)
    if np.isnan(dist):
        dist = _dist_lookup.get((destination, origin), np.nan)
    if np.isnan(dist):
        dist = float(np.median(list(_dist_lookup.values())))
    return dist


def _row(hour, month, day_of_week, origin, destination, airline_iata, distance):
    """Build a single-row feature DataFrame matching the trained model schema."""
    is_weekend = int(day_of_week in [6, 7])
    is_holiday = int((month, 15) in HOLIDAY_SET)

    if 5 <= hour <= 11:   tod = 0
    elif 12 <= hour <= 17: tod = 1
    elif 18 <= hour <= 21: tod = 2
    else:                  tod = 3

    dist_bin = 0 if distance < 500 else (1 if distance <= 1500 else 2)
    route    = f"{origin}_{destination}"

    return pd.DataFrame([{
        "hour":               float(hour),
        "time_of_day":        float(tod),
        "day_of_week":        float(day_of_week),
        "is_weekend":         float(is_weekend),
        "month":              float(month),
        "is_holiday":         float(is_holiday),
        "distance_bin":       float(dist_bin),
        "carrier_delay_rate": _carrier_rate.get(airline_iata, _overall),
        "origin_delay_rate":  _origin_rate.get(origin, _overall),
        "route_delay_rate":   _route_rate.get(route, _overall),
    }])


def _predict(row_df):
    return float(_model.predict_proba(row_df)[0, 1]) * 100


def _route_carriers(origin, destination, min_route_flights):
    """Return DataFrame of carriers with enough flights on origin→destination."""
    return (
        _train[
            (_train["ORIGIN_AIRPORT"] == origin) &
            (_train["DESTINATION_AIRPORT"] == destination)
        ]
        .groupby("AIRLINE")
        .agg(route_flights=("delayed", "count"))
        .query(f"route_flights >= {min_route_flights}")
        .reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. recommend_airlines
# ══════════════════════════════════════════════════════════════════════════════
def recommend_airlines(origin, destination, month, day_of_week, hour,
                       min_route_flights=10):
    """
    For the given route and time, predict delay probability for every carrier
    with >= min_route_flights historical flights on that route.

    Objective: minimise predicted delay probability
    Constraint: carrier_flights >= min_route_flights  (service frequency)

    Returns ALL feasible carriers sorted by lowest predicted delay %:
        carrier_iata, carrier_name, predicted_delay_pct, route_flights
    """
    _load()

    carriers = _route_carriers(origin, destination, min_route_flights)
    if carriers.empty:
        return []

    distance = _route_distance(origin, destination)
    results  = []

    for _, r in carriers.iterrows():
        iata = r["AIRLINE"]
        pct  = _predict(_row(hour, month, day_of_week, origin, destination, iata, distance))
        results.append({
            "carrier_iata":        iata,
            "carrier_name":        _airline_name.get(iata, iata),
            "predicted_delay_pct": round(pct, 1),
            "route_flights":       int(r["route_flights"]),
        })

    results.sort(key=lambda x: x["predicted_delay_pct"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. recommend_airport
# ══════════════════════════════════════════════════════════════════════════════
def recommend_airport(city_airports, destination, carrier, month, day_of_week, hour):
    """
    Given a list of origin airports in the same city, rank each by predicted
    delay probability.

    Returns a list of dicts (lowest delay first):
        airport_code, predicted_delay_pct, historical_ontime_pct
    """
    _load()

    results = []
    for ap in city_airports:
        distance    = _route_distance(ap, destination)
        pct         = _predict(_row(hour, month, day_of_week, ap, destination, carrier, distance))
        hist_ontime = (1 - _origin_rate.get(ap, _overall)) * 100
        results.append({
            "airport_code":          ap,
            "predicted_delay_pct":   round(pct, 1),
            "historical_ontime_pct": round(hist_ontime, 1),
        })

    results.sort(key=lambda x: x["predicted_delay_pct"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. optimize_flight
# ══════════════════════════════════════════════════════════════════════════════
def optimize_flight(origin, destination, month, day_of_week,
                    hour_min=5, hour_max=23,
                    risk_tolerance=30.0,
                    carrier_constraint=None,
                    min_route_flights=10):
    """
    Find the (carrier, hour) combination that minimises predicted delay
    probability subject to:
      - hour ∈ [hour_min, hour_max]
      - predicted delay % ≤ risk_tolerance
      - carrier has ≥ min_route_flights on the route
      - if carrier_constraint is set, restrict to that IATA code

    If no combination satisfies risk_tolerance, the constraint is relaxed
    and the overall best is returned with feasible=False.

    Returns:
        {
          feasible:     bool   — True if result is within risk_tolerance
          relaxed:      bool   — True if risk_tolerance was relaxed
          optimal:      dict   — best (carrier, hour) candidate
          alternatives: list   — up to 5 next-best candidates
        }
    """
    _load()

    carriers = _route_carriers(origin, destination, min_route_flights)
    if carrier_constraint:
        carriers = carriers[carriers["AIRLINE"] == carrier_constraint]

    if carriers.empty:
        return {"feasible": False, "relaxed": True, "optimal": None, "alternatives": []}

    distance   = _route_distance(origin, destination)
    candidates = []

    for _, r in carriers.iterrows():
        iata = r["AIRLINE"]
        for h in range(hour_min, hour_max + 1):
            pct = _predict(_row(h, month, day_of_week, origin, destination, iata, distance))
            candidates.append({
                "carrier_iata":        iata,
                "carrier_name":        _airline_name.get(iata, iata),
                "hour":                h,
                "hour_label":          _fmt_hour(h),
                "predicted_delay_pct": round(pct, 1),
                "route_flights":       int(r["route_flights"]),
            })

    candidates.sort(key=lambda x: x["predicted_delay_pct"])

    feasible = [c for c in candidates if c["predicted_delay_pct"] <= risk_tolerance]

    if feasible:
        return {
            "feasible":     True,
            "relaxed":      False,
            "optimal":      feasible[0],
            "alternatives": feasible[1:6],
        }
    else:
        return {
            "feasible":     False,
            "relaxed":      True,
            "optimal":      candidates[0] if candidates else None,
            "alternatives": candidates[1:6],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. sensitivity_analysis
# ══════════════════════════════════════════════════════════════════════════════
def sensitivity_analysis(origin, destination, month, day_of_week,
                         hour_min=5, hour_max=23, min_route_flights=10):
    """
    For risk tolerances 10 % → 50 % (step 5), count the number of feasible
    (carrier, hour) combinations on the route.

    Returns a list of {tolerance_pct, feasible_count}.
    """
    _load()

    carriers = _route_carriers(origin, destination, min_route_flights)
    if carriers.empty:
        return [{"tolerance_pct": t, "feasible_count": 0} for t in range(10, 55, 5)]

    distance = _route_distance(origin, destination)
    all_pcts = []

    for _, r in carriers.iterrows():
        iata = r["AIRLINE"]
        for h in range(hour_min, hour_max + 1):
            pct = _predict(_row(h, month, day_of_week, origin, destination, iata, distance))
            all_pcts.append(pct)

    return [
        {"tolerance_pct": t, "feasible_count": sum(1 for p in all_pcts if p <= t)}
        for t in range(10, 55, 5)
    ]


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== recommend_airlines (LAX→JFK, Jun, Mon, 8am, min_flights=10) ===")
    for r in recommend_airlines("LAX", "JFK", 6, 1, 8, min_route_flights=10):
        print(f"  {r['carrier_name']:<35} {r['predicted_delay_pct']:5.1f}%  "
              f"({r['route_flights']} flights)")

    print("\n=== recommend_airport (NYC → LAX, WN, Jun, Mon, 8am) ===")
    for r in recommend_airport(["JFK", "LGA", "EWR"], "LAX", "WN", 6, 1, 8):
        print(f"  {r['airport_code']}  delay={r['predicted_delay_pct']}%  "
              f"hist on-time={r['historical_ontime_pct']}%")

    print("\n=== optimize_flight (LAX→JFK, Jun, Mon, 7am-11am, risk≤25%) ===")
    res = optimize_flight("LAX", "JFK", 6, 1, hour_min=7, hour_max=11, risk_tolerance=25.0)
    if res["optimal"]:
        print(f"  Feasible: {res['feasible']}  Relaxed: {res['relaxed']}")
        o = res["optimal"]
        print(f"  Optimal: {o['carrier_name']} @ {o['hour_label']} → {o['predicted_delay_pct']}%")
        print(f"  Alternatives: {len(res['alternatives'])}")
    else:
        print("  No candidates found.")

    print("\n=== sensitivity_analysis (LAX→JFK, Jun, Mon, 7am-11am) ===")
    for row in sensitivity_analysis("LAX", "JFK", 6, 1, hour_min=7, hour_max=11):
        bar = "█" * row["feasible_count"]
        print(f"  {row['tolerance_pct']:>3}%  {row['feasible_count']:>3}  {bar}")

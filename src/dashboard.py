import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Allow importing optimizer from same directory
sys.path.insert(0, os.path.dirname(__file__))
from optimizer import (recommend_airlines, recommend_airport,
                        optimize_flight, sensitivity_analysis, _fmt_hour)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="US Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── Colour palette (blues) ────────────────────────────────────────────────────
BLUE_SEQ  = px.colors.sequential.Blues[3:]   # light→dark blues
BLUE_MAIN = "#2166ac"
BLUE_LIGHT = "#d1e5f0"
ACCENT    = "#d6604d"                        # red accent for delay highlights

MONTH_LABELS = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}
DAY_LABELS = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}

HOLIDAY_SET = frozenset([
    (1,1),(12,31),(1,2),(1,19),(2,16),(5,25),
    (7,3),(7,4),(7,5),(9,7),(11,25),(11,26),(11,27),
    (12,24),(12,25),(12,26),
])


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "flights_clean.csv"),
                     low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_train_rates():
    """Pre-compute delay rates from training split for the predictor."""
    from sklearn.model_selection import train_test_split
    df = load_data()
    train, _ = train_test_split(df, test_size=0.20, stratify=df["delayed"],
                                random_state=42)
    overall = train["delayed"].mean()
    carrier_rate = train.groupby("AIRLINE")["delayed"].mean().to_dict()
    origin_rate  = train.groupby("ORIGIN_AIRPORT")["delayed"].mean().to_dict()
    route_key    = train["ORIGIN_AIRPORT"].astype(str) + "_" + train["DESTINATION_AIRPORT"].astype(str)
    route_rate   = train.assign(_route=route_key).groupby("_route")["delayed"].mean().to_dict()
    return overall, carrier_rate, origin_rate, route_rate


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(os.path.join(ROOT, "models", "best_model_tuned.pkl"))


# ── Feature builder (mirrors feature_eng.py) ──────────────────────────────────
def build_features(hour, month, day_of_week, origin, destination, airline_iata,
                   distance, overall, carrier_rate, origin_rate, route_rate):
    is_weekend = int(day_of_week in [6, 7])
    is_holiday = int((month, 15) in HOLIDAY_SET)   # rough mid-month check

    if 5 <= hour <= 11:
        tod = 0   # morning
    elif 12 <= hour <= 17:
        tod = 1   # afternoon
    elif 18 <= hour <= 21:
        tod = 2   # evening
    else:
        tod = 3   # night

    if distance < 500:
        dist_bin = 0
    elif distance <= 1500:
        dist_bin = 1
    else:
        dist_bin = 2

    route = f"{origin}_{destination}"
    c_rate = carrier_rate.get(airline_iata, overall)
    o_rate = origin_rate.get(origin, overall)
    r_rate = route_rate.get(route, overall)

    return pd.DataFrame([{
        "hour":               float(hour),
        "time_of_day":        float(tod),
        "day_of_week":        float(day_of_week),
        "is_weekend":         float(is_weekend),
        "month":              float(month),
        "is_holiday":         float(is_holiday),
        "distance_bin":       float(dist_bin),
        "carrier_delay_rate": c_rate,
        "origin_delay_rate":  o_rate,
        "route_delay_rate":   r_rate,
    }])


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* main content padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* header */
    .dash-title  { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .dash-sub    { font-size: 1rem; color: #6b7280; margin-top: 0.1rem; margin-bottom: 0; }

    /* section headers */
    .section-header {
        font-size: 1.25rem; font-weight: 600; color: #1a1a2e;
        border-left: 4px solid #2166ac; padding-left: 10px;
        margin-bottom: 0.5rem;
    }

    /* predictor card */
    .pred-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 2rem;
    }

    /* footer */
    .footer { font-size: 0.78rem; color: #9ca3af; text-align: center; margin-top: 2rem; }

    /* optimizer airline card */
    .airline-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .airline-card .rank    { font-size: 0.75rem; color:#9ca3af; margin-bottom:2px; }
    .airline-card .name    { font-size: 0.95rem; font-weight:600; color:#1a1a2e; margin-bottom:6px; }
    .airline-card .pct     { font-size: 1.6rem; font-weight:700; }
    .airline-card .flights { font-size: 0.78rem; color:#6b7280; margin-top:4px; }
    .pct-low  { color: #16a34a; }
    .pct-med  { color: #d97706; }
    .pct-high { color: #dc2626; }

    /* hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# City → airport codes for Tab 2
CITY_AIRPORTS = {
    "New York":       ["JFK", "LGA", "EWR"],
    "Chicago":        ["ORD", "MDW"],
    "Los Angeles":    ["LAX", "SNA", "BUR"],
    "San Francisco":  ["SFO", "OAK", "SJC"],
    "Washington DC":  ["IAD", "DCA", "BWI"],
    "Dallas":         ["DFW", "DAL"],
    "Houston":        ["IAH", "HOU"],
}


# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df = load_data()
    overall_rate, carrier_rate, origin_rate, route_rate = load_train_rates()
    model = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="dash-title">✈️ US Flight Delay Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="dash-sub">Explore 2015 domestic flight reliability and predict your own delay risk.</p>',
            unsafe_allow_html=True)
st.divider()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Applied to Section 1 charts")

    # Carrier filter
    carrier_options = sorted(df["AIRLINE_NAME"].dropna().unique())
    sel_carriers = st.multiselect("Airline", carrier_options,
                                  placeholder="All airlines")

    # Airport filter
    airport_options = sorted(df["ORIGIN_AIRPORT"].dropna().unique())
    sel_airports = st.multiselect("Origin Airport (IATA)", airport_options,
                                  placeholder="All airports")

    st.divider()
    st.caption("Data: 250K sampled flights · 2015 · US DOT BTS")


# Apply sidebar filters
fdf = df.copy()
if sel_carriers:
    fdf = fdf[fdf["AIRLINE_NAME"].isin(sel_carriers)]
if sel_airports:
    fdf = fdf[fdf["ORIGIN_AIRPORT"].isin(sel_airports)]


# ── Top stats row ─────────────────────────────────────────────────────────────
total_flights = len(fdf)
on_time_pct   = (1 - fdf["delayed"].mean()) * 100

carrier_rates = (fdf.groupby("AIRLINE_NAME")["delayed"]
                    .mean().mul(100).sort_values(ascending=False))
worst_carrier = carrier_rates.index[0] if len(carrier_rates) else "N/A"
worst_carrier_pct = carrier_rates.iloc[0] if len(carrier_rates) else 0

airport_rates = (fdf.groupby("ORIGIN_AIRPORT")["delayed"]
                    .agg(["mean", "count"])
                    .query("count >= 50")["mean"]
                    .mul(100).sort_values(ascending=False))
worst_airport = airport_rates.index[0] if len(airport_rates) else "N/A"
worst_airport_pct = airport_rates.iloc[0] if len(airport_rates) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Flights",   f"{total_flights:,}")
c2.metric("On-Time Rate",    f"{on_time_pct:.1f}%",
          delta=f"{on_time_pct - 81.4:.1f}pp vs dataset avg",
          delta_color="normal")
c3.metric("Worst Carrier",   worst_carrier,
          delta=f"{worst_carrier_pct:.1f}% delayed", delta_color="inverse")
c4.metric("Worst Airport",   worst_airport,
          delta=f"{worst_airport_pct:.1f}% delayed", delta_color="inverse")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Flight Reliability Explorer
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 1 — Flight Reliability Explorer</p>',
            unsafe_allow_html=True)
st.markdown(" ")

# Row 1: Carrier bar + Hour line
row1_l, row1_r = st.columns(2)

with row1_l:
    data = (fdf.groupby("AIRLINE_NAME")["delayed"]
               .mean().mul(100)
               .sort_values(ascending=True)
               .reset_index()
               .rename(columns={"AIRLINE_NAME": "Airline", "delayed": "Delay %"}))
    fig = px.bar(data, x="Delay %", y="Airline", orientation="h",
                 title="Delay Rate by Airline",
                 color="Delay %", color_continuous_scale="Blues",
                 text=data["Delay %"].map(lambda v: f"{v:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title="% Delayed (≥15 min)", yaxis_title="",
                      margin=dict(l=0, r=20, t=40, b=0),
                      height=420, plot_bgcolor="white",
                      xaxis=dict(gridcolor="#e5e7eb"),
                      yaxis=dict(gridcolor="#e5e7eb"))
    fig.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

with row1_r:
    hour_data = fdf.copy()
    hour_data["hour"] = (hour_data["DEPARTURE_TIME"] // 100).astype("Int64")
    hour_data = (hour_data.dropna(subset=["hour"])
                          .loc[lambda d: d["hour"].between(0, 23)]
                          .groupby("hour")["delayed"]
                          .mean().mul(100)
                          .reset_index()
                          .rename(columns={"hour": "Hour", "delayed": "Delay %"}))
    hour_labels = [_fmt_hour(h) for h in range(24)]
    fig = px.line(hour_data, x="Hour", y="Delay %",
                  title="Delay Rate by Departure Hour",
                  markers=True,
                  color_discrete_sequence=[BLUE_MAIN])
    fig.update_traces(line_width=2.5)
    fig.update_layout(
        xaxis=dict(
            range=[-0.5, 23.5],
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=hour_labels,
            tickangle=-45,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
        plot_bgcolor="white",
        margin=dict(l=0, r=20, t=40, b=60),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 2: Month line + Top-15 airports bar
row2_l, row2_r = st.columns(2)

with row2_l:
    month_data = (fdf.groupby("MONTH")["delayed"]
                     .mean().mul(100)
                     .reset_index()
                     .rename(columns={"MONTH": "Month", "delayed": "Delay %"}))
    month_data["Month Label"] = month_data["Month"].map(MONTH_LABELS)
    fig = px.line(month_data, x="Month Label", y="Delay %",
                  title="Delay Rate by Month",
                  markers=True,
                  color_discrete_sequence=[BLUE_MAIN],
                  category_orders={"Month Label": list(MONTH_LABELS.values())})
    fig.update_traces(line_width=2.5)
    fig.update_layout(xaxis=dict(gridcolor="#e5e7eb"),
                      yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                      plot_bgcolor="white",
                      margin=dict(l=0, r=20, t=40, b=0),
                      height=380)
    st.plotly_chart(fig, use_container_width=True)

with row2_r:
    ap_data = (fdf.groupby("ORIGIN_AIRPORT_NAME")["delayed"]
                  .agg(["mean", "count"])
                  .query("count >= 50")["mean"]
                  .mul(100)
                  .sort_values(ascending=False)
                  .head(15)
                  .sort_values(ascending=True)
                  .reset_index()
                  .rename(columns={"ORIGIN_AIRPORT_NAME": "Airport", "mean": "Delay %"}))
    fig = px.bar(ap_data, x="Delay %", y="Airport", orientation="h",
                 title="Top 15 Origin Airports by Delay Rate",
                 color="Delay %", color_continuous_scale="Blues",
                 text=ap_data["Delay %"].map(lambda v: f"{v:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title="% Delayed", yaxis_title="",
                      xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                      yaxis=dict(gridcolor="#e5e7eb"),
                      plot_bgcolor="white",
                      margin=dict(l=0, r=20, t=40, b=0),
                      height=380)
    st.plotly_chart(fig, use_container_width=True)

# Row 3: Geo map (full width)
st.markdown(" ")
st.markdown('<p style="font-size:1rem;font-weight:600;color:#374151;">Delay Rates Across US Airports</p>',
            unsafe_allow_html=True)

geo_data = (fdf.groupby(["ORIGIN_AIRPORT", "ORIGIN_AIRPORT_NAME", "ORIGIN_LAT", "ORIGIN_LON"])
               .agg(delay_pct=("delayed", lambda x: x.mean() * 100),
                    flights=("delayed", "count"))
               .reset_index()
               .dropna(subset=["ORIGIN_LAT", "ORIGIN_LON"]))

fig_map = px.scatter_geo(
    geo_data,
    lat="ORIGIN_LAT", lon="ORIGIN_LON",
    color="delay_pct",
    size="flights",
    hover_name="ORIGIN_AIRPORT_NAME",
    hover_data={
        "delay_pct": ":.1f",
        "flights": ":,",
        "ORIGIN_LAT": False,
        "ORIGIN_LON": False,
    },
    labels={"delay_pct": "Delay %", "flights": "Flights"},
    color_continuous_scale="Blues",
    size_max=40,
    scope="usa",
    title="",
)
fig_map.update_layout(
    geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#f1f5f9",
             lakecolor="#dbeafe", coastlinecolor="#94a3b8",
             showland=True, showlakes=True, showcountries=False),
    coloraxis_colorbar=dict(title="Delay %", ticksuffix="%"),
    margin=dict(l=0, r=0, t=10, b=0),
    height=450,
    paper_bgcolor="white",
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Delay Predictor
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 2 — Delay Predictor</p>',
            unsafe_allow_html=True)
st.markdown(" ")

st.markdown('<div class="pred-card">', unsafe_allow_html=True)

pred_col, result_col = st.columns([1.1, 0.9], gap="large")

with pred_col:
    st.markdown("**Configure your flight**")

    # Build dropdown options from data
    airline_map = (df[["AIRLINE", "AIRLINE_NAME"]]
                   .dropna()
                   .drop_duplicates()
                   .sort_values("AIRLINE_NAME")
                   .set_index("AIRLINE_NAME")["AIRLINE"]
                   .to_dict())

    origins = sorted(df["ORIGIN_AIRPORT"].dropna().unique())
    dests   = sorted(df["DESTINATION_AIRPORT"].dropna().unique())

    inp1, inp2 = st.columns(2)
    with inp1:
        origin_sel = st.selectbox("Origin Airport", origins,
                                  index=origins.index("LAX") if "LAX" in origins else 0)
    with inp2:
        dest_sel = st.selectbox("Destination Airport", dests,
                                index=dests.index("JFK") if "JFK" in dests else 0)

    airline_sel_name = st.selectbox("Airline", list(airline_map.keys()),
                                    index=0)
    airline_iata = airline_map[airline_sel_name]

    # Try to infer typical distance for this route
    route_dist = df[
        (df["ORIGIN_AIRPORT"] == origin_sel) & (df["DESTINATION_AIRPORT"] == dest_sel)
    ]["DISTANCE"].median()
    if np.isnan(route_dist):
        route_dist = df["DISTANCE"].median()

    inp3, inp4, inp5 = st.columns(3)
    with inp3:
        month_sel = st.selectbox("Month",
                                 list(MONTH_LABELS.values()),
                                 index=5)
        month_num = {v: k for k, v in MONTH_LABELS.items()}[month_sel]
    with inp4:
        day_sel = st.selectbox("Day of Week",
                               list(DAY_LABELS.values()),
                               index=0)
        day_num = {v: k for k, v in DAY_LABELS.items()}[day_sel]
    with inp5:
        hour_sel = st.slider("Departure Hour", 0, 23, 8)

    predict_btn = st.button("✈️  Predict Delay Risk", type="primary",
                            use_container_width=True)

with result_col:
    st.markdown("**Prediction**")

    if predict_btn:
        X = build_features(
            hour=hour_sel, month=month_num, day_of_week=day_num,
            origin=origin_sel, destination=dest_sel,
            airline_iata=airline_iata, distance=route_dist,
            overall=overall_rate, carrier_rate=carrier_rate,
            origin_rate=origin_rate, route_rate=route_rate,
        )
        prob = model.predict_proba(X)[0, 1]
        pct  = prob * 100

        # Delay probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar":  {"color": ACCENT if pct > 40 else ("#f59e0b" if pct > 20 else BLUE_MAIN)},
                "steps": [
                    {"range": [0,  20], "color": "#d1fae5"},
                    {"range": [20, 40], "color": "#fef3c7"},
                    {"range": [40, 100],"color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 2},
                              "thickness": 0.75, "value": pct},
            },
            title={"text": "Delay Probability", "font": {"size": 14}},
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Reliability badge
        if pct < 20:
            st.success(f"Low risk — {pct:.1f}% chance of delay")
        elif pct < 40:
            st.warning(f"Moderate risk — {pct:.1f}% chance of delay")
        else:
            st.error(f"High risk — {pct:.1f}% chance of delay")

        # Comparison to route average
        route_key  = f"{origin_sel}_{dest_sel}"
        route_avg  = route_rate.get(route_key, overall_rate) * 100
        delta_vs_route = pct - route_avg
        sign = "+" if delta_vs_route >= 0 else ""
        st.caption(
            f"Route average: **{route_avg:.1f}%** &nbsp;|&nbsp; "
            f"Your flight: **{pct:.1f}%** &nbsp;({sign}{delta_vs_route:.1f}pp)"
        )

        # Top 3 risk factors
        st.markdown("**Top 3 risk factors**")
        factors = {
            "Route history":    X["route_delay_rate"].iloc[0],
            "Carrier history":  X["carrier_delay_rate"].iloc[0],
            "Departure hour":   X["hour"].iloc[0] / 23,        # normalised
            "Origin airport":   X["origin_delay_rate"].iloc[0],
        }
        top3 = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        for rank, (name, score) in enumerate(top3, 1):
            bar_width = int(score * 100)
            bar_color = ACCENT if score > 0.35 else ("#f59e0b" if score > 0.2 else BLUE_MAIN)
            st.markdown(
                f"**{rank}. {name}**  "
                f'<span style="display:inline-block;width:{bar_width}px;height:8px;'
                f'background:{bar_color};border-radius:4px;vertical-align:middle;'
                f'margin-left:8px;"></span>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Configure your flight on the left and click **Predict**.")

st.markdown("</div>", unsafe_allow_html=True)


st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Smart Booking Optimizer
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 3 — Smart Booking Optimizer</p>',
            unsafe_allow_html=True)
st.markdown('<p style="color:#6b7280;font-size:0.95rem;margin-bottom:1rem;">'
            'Find the lowest-risk way to fly.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Optimal Flight", "✈️  Best Airlines", "🏙️  Best Airport"])

# ── helpers shared across tabs ────────────────────────────────────────────────
_origins_list = sorted(df["ORIGIN_AIRPORT"].dropna().unique())
_dests_list   = sorted(df["DESTINATION_AIRPORT"].dropna().unique())
_airline_map  = (df[["AIRLINE", "AIRLINE_NAME"]].dropna().drop_duplicates()
                   .sort_values("AIRLINE_NAME")
                   .set_index("AIRLINE_NAME")["AIRLINE"].to_dict())


def _pct_class(pct):
    if pct < 20:  return "pct-low"
    if pct < 40:  return "pct-med"
    return "pct-high"


def _badge(pct):
    if pct < 20:  return "🟢"
    if pct < 40:  return "🟡"
    return "🔴"


# ── TAB 1: Optimal Flight Finder ──────────────────────────────────────────────
with tab1:
    st.markdown(" ")
    t1c1, t1c2 = st.columns(2)
    with t1c1:
        t1_origin = st.selectbox("Origin Airport", _origins_list,
                                 index=_origins_list.index("LAX") if "LAX" in _origins_list else 0,
                                 key="t1_origin")
    with t1c2:
        t1_dest = st.selectbox("Destination Airport", _dests_list,
                               index=_dests_list.index("JFK") if "JFK" in _dests_list else 0,
                               key="t1_dest")

    t1c3, t1c4 = st.columns(2)
    with t1c3:
        t1_month_label = st.selectbox("Month", list(MONTH_LABELS.values()),
                                      index=5, key="t1_month")
        t1_month = {v: k for k, v in MONTH_LABELS.items()}[t1_month_label]
    with t1c4:
        t1_day_label = st.selectbox("Day of Week", list(DAY_LABELS.values()),
                                    index=0, key="t1_day")
        t1_day = {v: k for k, v in DAY_LABELS.items()}[t1_day_label]

    t1_window = st.slider("Preferred departure window", 5, 23, (7, 21), key="t1_window")
    t1_hour_min, t1_hour_max = t1_window
    st.caption(f"Window: {_fmt_hour(t1_hour_min)} – {_fmt_hour(t1_hour_max)}")

    t1sl1, t1sl2 = st.columns(2)
    with t1sl1:
        t1_risk = st.slider("Risk tolerance (%)", 10, 50, 30, step=5, key="t1_risk")
    with t1sl2:
        t1_carrier_options = ["Any"] + list(_airline_map.keys())
        t1_carrier_label   = st.selectbox("Carrier constraint", t1_carrier_options,
                                          index=0, key="t1_carrier_constraint")
        t1_carrier_iata    = (None if t1_carrier_label == "Any"
                              else _airline_map[t1_carrier_label])

    if st.button("Optimize", type="primary", key="btn_t1"):
        with st.spinner("Searching for optimal flight…"):
            result = optimize_flight(
                t1_origin, t1_dest, t1_month, t1_day,
                hour_min=t1_hour_min, hour_max=t1_hour_max,
                risk_tolerance=float(t1_risk),
                carrier_constraint=t1_carrier_iata,
            )
            sens   = sensitivity_analysis(
                t1_origin, t1_dest, t1_month, t1_day,
                hour_min=t1_hour_min, hour_max=t1_hour_max,
            )

        if result["optimal"] is None:
            st.error("No carriers with sufficient historical data on this route. Try a busier route.")
        else:
            opt = result["optimal"]
            # Feasibility badge
            if result["feasible"]:
                st.success(
                    f"✅ Feasible solution found — **{opt['carrier_name']}** "
                    f"departing at **{opt['hour_label']}** with "
                    f"**{opt['predicted_delay_pct']:.1f}%** predicted delay risk "
                    f"(within your {t1_risk}% tolerance)"
                )
            else:
                st.warning(
                    f"⚠️ No option within {t1_risk}% risk tolerance — showing best available: "
                    f"**{opt['carrier_name']}** at **{opt['hour_label']}** "
                    f"({opt['predicted_delay_pct']:.1f}% delay risk). "
                    f"Try raising the tolerance slider."
                )

            # Optimal card + alternatives side by side
            opt_col, alt_col = st.columns([1, 1.6], gap="large")

            with opt_col:
                st.markdown("**Optimal choice**")
                cls = _pct_class(opt["predicted_delay_pct"])
                st.markdown(f"""
<div class="airline-card" style="text-align:left;padding:1.2rem;">
  <div class="rank">🎯 Best match</div>
  <div class="name" style="font-size:1.05rem;">{opt['carrier_name']}</div>
  <div style="margin:6px 0;color:#6b7280;font-size:0.85rem;">
    Departure: <strong>{opt['hour_label']}</strong> &nbsp;|&nbsp;
    {opt['route_flights']:,} route flights
  </div>
  <div class="pct {cls}" style="font-size:2rem;">{_badge(opt['predicted_delay_pct'])} {opt['predicted_delay_pct']:.1f}%</div>
</div>""", unsafe_allow_html=True)

            with alt_col:
                if result["alternatives"]:
                    st.markdown("**Top alternatives**")
                    alt_rows = []
                    for i, a in enumerate(result["alternatives"], 1):
                        alt_rows.append({
                            "Rank": f"#{i+1}",
                            "Airline":      a["carrier_name"],
                            "Departure":    a["hour_label"],
                            "Delay Risk":   f"{a['predicted_delay_pct']:.1f}%",
                            "Route Flights": f"{a['route_flights']:,}",
                        })
                    alt_df = pd.DataFrame(alt_rows)

                    # Colour the delay column
                    def _colour_delay(val):
                        pct = float(val.replace("%", ""))
                        if pct < 20:   color = "#d1fae5"
                        elif pct < 40: color = "#fef3c7"
                        else:          color = "#fee2e2"
                        return f"background-color: {color}"

                    styled = alt_df.style.map(_colour_delay, subset=["Delay Risk"])
                    st.dataframe(styled, hide_index=True, use_container_width=True)
                else:
                    st.info("No alternatives within the search space.")

            # Sensitivity analysis chart
            st.markdown(" ")
            with st.expander("Sensitivity: risk tolerance vs number of feasible options"):
                sens_df = pd.DataFrame(sens)
                current_count = next(
                    (r["feasible_count"] for r in sens if r["tolerance_pct"] == t1_risk), 0
                )
                fig_sens = px.bar(
                    sens_df, x="tolerance_pct", y="feasible_count",
                    title="Feasible (carrier, hour) combinations at each risk tolerance",
                    color="feasible_count",
                    color_continuous_scale="Blues",
                    text=sens_df["feasible_count"],
                    labels={"tolerance_pct": "Risk Tolerance (%)", "feasible_count": "# Options"},
                )
                fig_sens.add_vline(x=t1_risk, line_dash="dash",
                                   line_color=ACCENT, line_width=2,
                                   annotation_text=f"Current: {t1_risk}% → {current_count} options",
                                   annotation_font_color=ACCENT,
                                   annotation_position="top right")
                fig_sens.update_traces(textposition="outside")
                fig_sens.update_layout(
                    coloraxis_showscale=False, plot_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%",
                               tickvals=[r["tolerance_pct"] for r in sens]),
                    yaxis=dict(gridcolor="#e5e7eb"),
                    margin=dict(t=50, b=10), height=320,
                )
                st.plotly_chart(fig_sens, use_container_width=True)
                st.caption(
                    "Each bar counts distinct (airline × departure hour) combinations "
                    "on this route whose predicted delay % falls within that tolerance."
                )


# ── TAB 2: Best Airlines ──────────────────────────────────────────────────────
with tab2:
    st.markdown(" ")
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        t2_origin = st.selectbox("Origin Airport", _origins_list,
                                 index=_origins_list.index("LAX") if "LAX" in _origins_list else 0,
                                 key="t2_origin")
    with t2c2:
        t2_dest = st.selectbox("Destination Airport", _dests_list,
                               index=_dests_list.index("JFK") if "JFK" in _dests_list else 0,
                               key="t2_dest")

    t2c3, t2c4, t2c5 = st.columns(3)
    with t2c3:
        t2_month_label = st.selectbox("Month", list(MONTH_LABELS.values()),
                                      index=5, key="t2_month")
        t2_month = {v: k for k, v in MONTH_LABELS.items()}[t2_month_label]
    with t2c4:
        t2_day_label = st.selectbox("Day of Week", list(DAY_LABELS.values()),
                                    index=0, key="t2_day")
        t2_day = {v: k for k, v in DAY_LABELS.items()}[t2_day_label]
    with t2c5:
        t2_hour = st.slider("Departure Hour", 5, 23, 8, key="t2_hour", format="%d:00")

    t2_min_flights = st.slider("Minimum flight frequency on route", 5, 50, 10, step=5,
                               key="t2_min_flights",
                               help="Only show carriers with at least this many historical flights on the route")

    if st.button("Find Best Airlines", type="primary", key="btn_t2"):
        with st.spinner("Predicting…"):
            recs = recommend_airlines(t2_origin, t2_dest, t2_month, t2_day, t2_hour,
                                      min_route_flights=t2_min_flights)

        if not recs:
            st.warning(
                f"No carriers found with ≥ {t2_min_flights} flights on "
                f"{t2_origin} → {t2_dest}. Try lowering the minimum flight frequency."
            )
        else:
            st.markdown(" ")
            # Full ranked table
            table_df = pd.DataFrame(recs)[
                ["carrier_name", "predicted_delay_pct", "route_flights", "carrier_iata"]
            ].rename(columns={
                "carrier_name":        "Airline",
                "predicted_delay_pct": "Predicted Delay %",
                "route_flights":       "Route Flights",
                "carrier_iata":        "IATA",
            })
            table_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(table_df))])
            table_df["Predicted Delay %"] = table_df["Predicted Delay %"].map(lambda v: f"{v:.1f}%")
            table_df["Route Flights"]     = table_df["Route Flights"].map(lambda v: f"{v:,}")

            def _row_colour(row):
                pct = float(row["Predicted Delay %"].replace("%", ""))
                bg  = "#d1fae5" if pct < 20 else ("#fef3c7" if pct < 40 else "#fee2e2")
                return [f"background-color: {bg}"] * len(row)

            styled_tbl = table_df.style.apply(_row_colour, axis=1)
            st.dataframe(styled_tbl, hide_index=True, use_container_width=True)

            st.caption(
                f"**Objective:** minimise predicted delay probability &nbsp;|&nbsp; "
                f"**Constraint:** ≥ {t2_min_flights} historical flights on {t2_origin} → {t2_dest} "
                f"&nbsp;|&nbsp; {len(recs)} carrier{'s' if len(recs) != 1 else ''} qualify"
            )

            # Bar chart
            bar_df = pd.DataFrame(recs).rename(
                columns={"carrier_name": "Airline", "predicted_delay_pct": "Predicted Delay %"})
            fig = px.bar(bar_df, x="Airline", y="Predicted Delay %",
                         color="Predicted Delay %", color_continuous_scale="Blues",
                         text=bar_df["Predicted Delay %"].map(lambda v: f"{v:.1f}%"),
                         title=f"Delay Risk by Carrier — {t2_origin} → {t2_dest}  "
                               f"({t2_month_label}, {t2_day_label}, {_fmt_hour(t2_hour)})")
            fig.update_traces(textposition="outside")
            fig.update_layout(
                coloraxis_showscale=False, plot_bgcolor="white",
                yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%",
                           range=[0, max(r["predicted_delay_pct"] for r in recs) * 1.3]),
                xaxis=dict(gridcolor="#e5e7eb"),
                margin=dict(t=50, b=10), height=320,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 3: Best Airport ───────────────────────────────────────────────────────
with tab3:
    st.markdown(" ")
    t3c1, t3c2 = st.columns(2)
    with t3c1:
        t3_city = st.selectbox("City (origin)", list(CITY_AIRPORTS.keys()), key="t3_city")
        t3_airports = CITY_AIRPORTS[t3_city]
        st.caption(f"Airports: {', '.join(t3_airports)}")
    with t3c2:
        t3_dest = st.selectbox("Destination Airport", _dests_list,
                               index=_dests_list.index("LAX") if "LAX" in _dests_list else 0,
                               key="t3_dest")

    t3_carrier_name = st.selectbox("Airline", list(_airline_map.keys()),
                                   index=0, key="t3_carrier")
    t3_carrier = _airline_map[t3_carrier_name]

    t3c3, t3c4, t3c5 = st.columns(3)
    with t3c3:
        t3_month_label = st.selectbox("Month", list(MONTH_LABELS.values()),
                                      index=5, key="t3_month")
        t3_month = {v: k for k, v in MONTH_LABELS.items()}[t3_month_label]
    with t3c4:
        t3_day_label = st.selectbox("Day of Week", list(DAY_LABELS.values()),
                                    index=0, key="t3_day")
        t3_day = {v: k for k, v in DAY_LABELS.items()}[t3_day_label]
    with t3c5:
        t3_hour = st.slider("Departure Hour", 5, 23, 8, key="t3_hour", format="%d:00")

    if st.button("Compare Airports", type="primary", key="btn_t3"):
        with st.spinner("Predicting…"):
            recs = recommend_airport(t3_airports, t3_dest, t3_carrier,
                                     t3_month, t3_day, t3_hour)

        st.markdown(" ")
        mcols = st.columns(len(recs))
        for i, (col, rec) in enumerate(zip(mcols, recs)):
            delta_txt = ("Best choice ✓" if i == 0
                         else f"+{rec['predicted_delay_pct'] - recs[0]['predicted_delay_pct']:.1f}pp vs best")
            col.metric(
                label=rec["airport_code"],
                value=f"{rec['predicted_delay_pct']:.1f}%",
                delta=delta_txt,
                delta_color="off" if i == 0 else "inverse",
            )

        chart_df = pd.DataFrame(recs).rename(
            columns={"airport_code": "Airport", "predicted_delay_pct": "Predicted Delay %"})
        fig = px.bar(chart_df, x="Airport", y="Predicted Delay %",
                     color="Predicted Delay %", color_continuous_scale="Blues",
                     text=chart_df["Predicted Delay %"].map(lambda v: f"{v:.1f}%"),
                     title=f"Predicted Delay % by Origin Airport — to {t3_dest}  "
                           f"({t3_carrier_name}, {t3_month_label}, {t3_day_label})")
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False, plot_bgcolor="white",
            yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%",
                       range=[0, max(r["predicted_delay_pct"] for r in recs) * 1.3]),
            xaxis=dict(gridcolor="#e5e7eb"),
            margin=dict(t=50, b=10), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Historical on-time rates (training data): " +
                   "  |  ".join(f"**{r['airport_code']}** {r['historical_ontime_pct']:.1f}%"
                                for r in recs))


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(" ")
st.markdown(
    '<p class="footer">Data: US DOT Bureau of Transportation Statistics, 2015 · '
    'Model: XGBoost (tuned) · Built with Streamlit</p>',
    unsafe_allow_html=True,
)

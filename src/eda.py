import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flights_clean.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

sns.set_theme(style="whitegrid")
FIGSIZE = (10, 6)
DELAY_COL = "delayed"


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_delay_by_carrier(df):
    carrier_col = "AIRLINE_NAME" if "AIRLINE_NAME" in df.columns else "AIRLINE"
    pct = df.groupby(carrier_col)[DELAY_COL].mean().mul(100).sort_values()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = sns.color_palette("Blues_d", len(pct))
    ax.barh(pct.index, pct.values, color=colors)
    ax.set_xlabel("% Flights Delayed (≥15 min)")
    ax.set_title("Delay Rate by Airline")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    save(fig, "delay_by_carrier.png")

    top = pct.idxmax()
    bot = pct.idxmin()
    print(f"[delay_by_carrier] {top} has the highest delay rate ({pct.max():.1f}%); "
          f"{bot} is the most on-time ({pct.min():.1f}%).")


def plot_delay_by_hour(df):
    df = df.copy()
    df["HOUR"] = (df["DEPARTURE_TIME"] // 100).astype("Int64")
    df = df.dropna(subset=["HOUR"])
    pct = df.groupby("HOUR")[DELAY_COL].mean().mul(100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(pct.index, pct.values, marker="o", linewidth=2, color=sns.color_palette("Blues_d")[3])
    ax.set_xlabel("Departure Hour (24h)")
    ax.set_ylabel("% Delayed")
    ax.set_title("Delay Rate by Departure Hour")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xticks(range(0, 24))
    save(fig, "delay_by_hour.png")

    peak_h = pct.idxmax()
    best_h = pct.idxmin()
    print(f"[delay_by_hour] Delays peak around {peak_h:02d}:00 ({pct.max():.1f}% delayed); "
          f"early morning ({best_h:02d}:00) has the lowest rate ({pct.min():.1f}%).")


def plot_delay_by_month(df):
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pct = df.groupby("MONTH")[DELAY_COL].mean().mul(100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(pct.index, pct.values, marker="o", linewidth=2, color=sns.color_palette("Blues_d")[3])
    ax.set_xticks(pct.index)
    ax.set_xticklabels([month_names[m] for m in pct.index])
    ax.set_ylabel("% Delayed")
    ax.set_title("Delay Rate by Month")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    save(fig, "delay_by_month.png")

    peak_m = pct.idxmax()
    best_m = pct.idxmin()
    print(f"[delay_by_month] {month_names[peak_m]} has the most delays ({pct.max():.1f}%); "
          f"{month_names[best_m]} is the most reliable ({pct.min():.1f}%).")


def plot_delay_by_day(df):
    day_names = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
    pct = df.groupby("DAY_OF_WEEK")[DELAY_COL].mean().mul(100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [sns.color_palette("Blues_d")[4] if v == pct.max() else sns.color_palette("Blues")[2]
              for v in pct.values]
    ax.bar([day_names.get(d, d) for d in pct.index], pct.values, color=colors)
    ax.set_ylabel("% Delayed")
    ax.set_title("Delay Rate by Day of Week")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    save(fig, "delay_by_day.png")

    peak_d = day_names.get(pct.idxmax(), pct.idxmax())
    best_d = day_names.get(pct.idxmin(), pct.idxmin())
    print(f"[delay_by_day] {peak_d} is the worst day to fly ({pct.max():.1f}% delayed); "
          f"{best_d} is the smoothest ({pct.min():.1f}%).")


def plot_delay_distribution(df):
    clipped = df["ARRIVAL_DELAY"].clip(-30, 120)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(clipped, bins=60, color=sns.color_palette("Blues_d")[3], edgecolor="white", linewidth=0.4)
    ax.axvline(15, color="tomato", linestyle="--", linewidth=1.5, label="15-min threshold")
    ax.set_xlabel("Arrival Delay (minutes, clipped −30 to 120)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Arrival Delay")
    ax.legend()
    save(fig, "delay_distribution.png")

    median_d = df["ARRIVAL_DELAY"].median()
    pct_pos = (df["ARRIVAL_DELAY"] > 0).mean() * 100
    print(f"[delay_distribution] Median arrival delay is {median_d:.1f} min; "
          f"{pct_pos:.1f}% of flights land late (any delay > 0).")


def plot_top_airports_delay(df):
    airport_col = "ORIGIN_AIRPORT_NAME" if "ORIGIN_AIRPORT_NAME" in df.columns else "ORIGIN_AIRPORT"
    stats = df.groupby(airport_col)[DELAY_COL].agg(["mean", "count"])
    stats = stats[stats["count"] >= 50]  # require enough samples
    top20 = stats["mean"].mul(100).sort_values(ascending=False).head(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Blues_d", len(top20))
    ax.barh(top20.index, top20.values, color=colors)
    ax.set_xlabel("% Flights Delayed (≥15 min)")
    ax.set_title("Top 20 Origin Airports by Delay Rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    save(fig, "top_airports_delay.png")

    worst = top20.idxmax()
    print(f"[top_airports_delay] {worst} tops the list with a {top20.max():.1f}% delay rate "
          f"among the 20 worst-performing origin airports.")


def plot_class_balance(df):
    counts = df[DELAY_COL].value_counts()
    labels = ["On-time", "Delayed"]
    sizes = [counts.get(0, 0), counts.get(1, 0)]
    colors = [sns.color_palette("Blues")[2], sns.color_palette("Blues_d")[4]]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(12)
    ax.set_title("Class Balance: Delayed vs On-Time")
    save(fig, "class_balance.png")

    delay_pct = counts.get(1, 0) / len(df) * 100
    print(f"[class_balance] {delay_pct:.1f}% of flights are delayed (≥15 min); "
          f"dataset is {'moderately' if 30 <= delay_pct <= 50 else 'imbalanced'} — "
          f"consider class weighting during modelling.")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(df):,} rows, {df.shape[1]} columns\n")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("1/7 delay_by_carrier.png")
    plot_delay_by_carrier(df)

    print("\n2/7 delay_by_hour.png")
    plot_delay_by_hour(df)

    print("\n3/7 delay_by_month.png")
    plot_delay_by_month(df)

    print("\n4/7 delay_by_day.png")
    plot_delay_by_day(df)

    print("\n5/7 delay_distribution.png")
    plot_delay_distribution(df)

    print("\n6/7 top_airports_delay.png")
    plot_top_airports_delay(df)

    print("\n7/7 class_balance.png")
    plot_class_balance(df)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()

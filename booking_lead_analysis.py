import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Fixed data path ---
DATA_DIR = Path("/Users/zaklelex/Desktop/BM_data/")
INPUT_FILE = DATA_DIR / "the-brass-monkey-kansas-city-38420-Transactions-2025-09-01-2025-10-21.csv"  # rename as needed
OUTPUT_DIR = DATA_DIR / "booking_outputs"
TARGET_WEEKDAY = 2  # Wednesday
TODAY = datetime.today().date()

# --- Core logic ---
def load_and_clean(df):
    df = df[df["Action"].astype(str).str.upper().str.strip() == "BOOKED"].copy()
    df["transaction_date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.date
    df["booking_date"] = pd.to_datetime(df["Booking Date"], errors="coerce").dt.date
    df.dropna(subset=["transaction_date", "booking_date"], inplace=True)
    df["days_in_advance"] = (pd.to_datetime(df["booking_date"]) - pd.to_datetime(df["transaction_date"])).dt.days
    df["qty"] = 1
    return df[["transaction_date", "booking_date", "days_in_advance", "qty"]]


def build_daily_snapshot(booked_df):
    snap = (
        booked_df.groupby(["transaction_date", "booking_date"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "bookings"})
    )
    snap["days_in_advance"] = (
        pd.to_datetime(snap["booking_date"]) - pd.to_datetime(snap["transaction_date"])
    ).dt.days
    return snap.sort_values(["transaction_date", "booking_date"]).reset_index(drop=True)


def build_cumulative_asof_snapshot(daily_snapshot):
    results = []
    for bdate, group in daily_snapshot.groupby("booking_date"):
        g = group.groupby("transaction_date", as_index=False)["bookings"].sum()
        start = g["transaction_date"].min()
        end = bdate
        if start is None:
            continue
        date_range = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D").date
        g_idx = g.set_index("transaction_date").reindex(date_range, fill_value=0)
        g_idx.index.name = "as_of_date"
        g_idx = g_idx.reset_index()
        g_idx["cum_bookings"] = g_idx["bookings"].cumsum()
        g_idx["booking_date"] = bdate
        g_idx["lookahead_days"] = (pd.to_datetime(g_idx["booking_date"]) - pd.to_datetime(g_idx["as_of_date"])).dt.days
        results.append(g_idx[["booking_date", "as_of_date", "lookahead_days", "cum_bookings"]])
    return pd.concat(results, ignore_index=True)


def compare_week_over_week(cum_df, as_of_date, target_weekday):
    start_of_week = as_of_date - timedelta(days=as_of_date.weekday())
    this_target = start_of_week + timedelta(days=target_weekday)
    last_target = this_target - timedelta(days=7)
    last_as_of = as_of_date - timedelta(days=7)

    def get_cum(booking_date, ref_as_of):
        row = cum_df[(cum_df["booking_date"] == booking_date) & (cum_df["as_of_date"] == ref_as_of)]
        return int(row["cum_bookings"].iloc[0]) if not row.empty else None

    this_val = get_cum(this_target, as_of_date)
    last_val = get_cum(last_target, last_as_of)
    return pd.DataFrame(
        [{
            "as_of_date": as_of_date,
            "this_target": this_target,
            "last_target": last_target,
            "this_week_bookings": this_val,
            "last_week_bookings": last_val,
            "delta": None if this_val is None or last_val is None else this_val - last_val
        }]
    )


def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    booked = load_and_clean(df)
    daily = build_daily_snapshot(booked)
    cum = build_cumulative_asof_snapshot(daily)

    daily.to_csv(OUTPUT_DIR / "daily_snapshot.csv", index=False)
    cum.to_csv(OUTPUT_DIR / "cumulative_asof_snapshot.csv", index=False)

    latest_as_of = cum["as_of_date"].max()
    wow = compare_week_over_week(cum, latest_as_of, TARGET_WEEKDAY)
    wow.to_csv(OUTPUT_DIR / "week_over_week_latest.csv", index=False)

    print("Analysis complete.")
    print(f"Daily snapshot: {OUTPUT_DIR}/daily_snapshot.csv")
    print(f"Cumulative as-of: {OUTPUT_DIR}/cumulative_asof_snapshot.csv")
    print(f"Week-over-week: {OUTPUT_DIR}/week_over_week_latest.csv")


if __name__ == "__main__":
    run_analysis()

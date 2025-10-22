
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Fixed data path ---
DATA_DIR = Path("/Users/zaklelex/Desktop/BM_data/")
INPUT_FILE = DATA_DIR / "the-brass-monkey-kansas-city-38420-Transactions-2025-09-01-2025-10-21.csv"  # adjust filename as needed
OUTPUT_DIR = DATA_DIR / "booking_outputs"
TARGET_WEEKDAY = 2  # Wednesday
TODAY = datetime.today().date()

# Candidate columns for party size / guests
PARTY_SIZE_CANDIDATES = [
    "Party Size", "Party size", "party size",
    "Guests", "Guest Count", "guest count",
    "Covers", "Cover Count", "covers", "covers count",
    "Qty", "Quantity", "quantity"
]

def _coerce_party_size(df: pd.DataFrame) -> pd.Series:
    # Find the first matching party size column
    for col in PARTY_SIZE_CANDIDATES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            # Replace missing or invalid with 1 as a safe default
            s = s.fillna(1).astype(int)
            # Clip at minimum 1
            s = s.clip(lower=1)
            return s
    # Fallback if no column found
    return pd.Series([1] * len(df), index=df.index, name="party_size")

def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only BOOKED
    df = df[df["Action"].astype(str).str.upper().str.strip() == "BOOKED"].copy()

    # Dates
    df["transaction_date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.date
    df["booking_date"] = pd.to_datetime(df["Booking Date"], errors="coerce").dt.date
    df = df.dropna(subset=["transaction_date", "booking_date"]).copy()

    # Lead days
    df["days_in_advance"] = (pd.to_datetime(df["booking_date"]) - pd.to_datetime(df["transaction_date"])).dt.days

    # Party size / guests
    df["guests"] = _coerce_party_size(df)
    # Bookings count is 1 per booking row
    df["bookings"] = 1

    return df[["transaction_date", "booking_date", "days_in_advance", "bookings", "guests"]]

def build_daily_snapshot(booked_df: pd.DataFrame) -> pd.DataFrame:
    snap = (
        booked_df.groupby(["transaction_date", "booking_date"], as_index=False)[["bookings", "guests"]]
        .sum()
    )
    snap["days_in_advance"] = (
        pd.to_datetime(snap["booking_date"]) - pd.to_datetime(snap["transaction_date"])
    ).dt.days
    return snap.sort_values(["transaction_date", "booking_date"]).reset_index(drop=True)

def build_cumulative_asof_snapshot(daily_snapshot: pd.DataFrame) -> pd.DataFrame:
    results = []
    for bdate, group in daily_snapshot.groupby("booking_date"):
        g = group.groupby("transaction_date", as_index=False)[["bookings", "guests"]].sum()
        start = g["transaction_date"].min()
        end = bdate
        if pd.isna(start):
            continue
        date_range = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D").date
        g_idx = g.set_index("transaction_date").reindex(date_range, fill_value=0)
        g_idx.index.name = "as_of_date"
        g_idx = g_idx.reset_index()
        g_idx["cum_bookings"] = g_idx["bookings"].cumsum()
        g_idx["cum_guests"] = g_idx["guests"].cumsum()
        g_idx["booking_date"] = bdate
        g_idx["lookahead_days"] = (pd.to_datetime(g_idx["booking_date"]) - pd.to_datetime(g_idx["as_of_date"])).dt.days
        results.append(g_idx[["booking_date", "as_of_date", "lookahead_days", "cum_bookings", "cum_guests"]])
    if results:
        out = pd.concat(results, ignore_index=True)
        out["booking_date"] = pd.to_datetime(out["booking_date"]).dt.date
        out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.date
        return out.sort_values(["booking_date", "as_of_date"]).reset_index(drop=True)
    return pd.DataFrame(columns=["booking_date", "as_of_date", "lookahead_days", "cum_bookings", "cum_guests"])

def compare_week_over_week(cum_df: pd.DataFrame, as_of_date, target_weekday: int) -> pd.DataFrame:
    start_of_week = as_of_date - timedelta(days=as_of_date.weekday())
    this_target = start_of_week + timedelta(days=target_weekday)
    last_target = this_target - timedelta(days=7)
    last_as_of = as_of_date - timedelta(days=7)

    def get_vals(bdate, ref_as_of):
        row = cum_df[(cum_df["booking_date"] == bdate) & (cum_df["as_of_date"] == ref_as_of)]
        if row.empty:
            return None, None
        return int(row["cum_bookings"].iloc[0]), int(row["cum_guests"].iloc[0])

    b_this, g_this = get_vals(this_target, as_of_date)
    b_last, g_last = get_vals(last_target, last_as_of)

    return pd.DataFrame([{
        "as_of_date": as_of_date,
        "this_target": this_target,
        "last_target": last_target,
        "this_week_bookings": b_this,
        "last_week_bookings": b_last,
        "delta_bookings": None if b_this is None or b_last is None else b_this - b_last,
        "this_week_guests": g_this,
        "last_week_guests": g_last,
        "delta_guests": None if g_this is None or g_last is None else g_this - g_last,
    }])

def run_analysis_local():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load CSV from the fixed path
    df = pd.read_csv(INPUT_FILE)
    booked = load_and_clean(df)
    daily = build_daily_snapshot(booked)
    cum = build_cumulative_asof_snapshot(daily)
    daily.to_csv(OUTPUT_DIR / "daily_snapshot.csv", index=False)
    cum.to_csv(OUTPUT_DIR / "cumulative_asof_snapshot.csv", index=False)
    # Use the latest as_of_date present
    if not cum.empty:
        latest_as_of = cum["as_of_date"].max()
        wow = compare_week_over_week(cum, latest_as_of, TARGET_WEEKDAY)
        wow.to_csv(OUTPUT_DIR / "week_over_week_latest.csv", index=False)
    print("Saved outputs to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_analysis_local()

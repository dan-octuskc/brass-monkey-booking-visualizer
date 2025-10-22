
#!/usr/bin/env python3
"""
booking_lead_analysis2.py
Adds Instagram marketing data into the daily bookings pipeline by matching on DATE.

What it produces (under booking_outputs/):
- marketing_vs_bookings.csv : one row per calendar date with bookings added that day and IG metrics (incl. lags & 7d windows)
- marketing_corr.csv        : quick correlation table vs bookings/guests

Inputs it expects:
- booking_outputs/daily_snapshot.csv  (from the prior pipeline)
    Must include at least: transaction_date, bookings, guests
- Insta_data.csv (or any file matching "*insta*.csv" / "*instagram*.csv") under the base folder

Usage (defaults to Dan's base folder):
    python booking_lead_analysis2.py
Or specify a different base folder:
    python booking_lead_analysis2.py /path/to/BM_data
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

def find_instagram_csv(base: Path) -> Optional[Path]:
    # try common names in base and subfolders (depth 2)
    candidates: List[Path] = []
    patterns = ["*Insta*.csv", "*insta*.csv", "*Instagram*.csv", "*instagram*.csv"]
    for pat in patterns:
        candidates.extend(list(base.glob(pat)))
        candidates.extend(list((base / "marketing").glob(pat)))
        candidates.extend(list((base / "data").glob(pat)))
    # De-dup, prefer names starting with "Insta"
    uniq = []
    seen = set()
    for p in candidates:
        if p.exists() and p.is_file() and p.suffix.lower() == ".csv":
            if p.resolve() not in seen:
                uniq.append(p)
                seen.add(p.resolve())
    if not uniq:
        return None
    # pick the shortest name or first
    uniq.sort(key=lambda p: (0 if p.name.lower().startswith("insta") else 1, len(p.name)))
    return uniq[0]

def parse_instagram_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [re.sub(r"\s+", "_", c.strip()).lower() for c in df.columns]
    # Expect a date col
    date_col = None
    for c in df.columns:
        if c in ("date", "day", "day_date", "dt"):
            date_col = c
            break
    if date_col is None:
        # Try to detect by dtype or first column
        date_col = df.columns[0]
    # Parse dates (MM/DD/YY or ISO)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    # Rename metrics to canonical names
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        c0 = c.replace("_", "").lower()
        if c0 in ("instagramclicks", "igclicks", "clicks"):
            rename_map[c] = "ig_clicks"
        elif c0 in ("instagramreach", "igreach", "reach"):
            rename_map[c] = "ig_reach"
        elif c0 in ("impressions", "instagramimpressions", "igimpressions"):
            rename_map[c] = "ig_impressions"
        elif c0 in ("profilevisits", "profile_visits"):
            rename_map[c] = "ig_profile_visits"
        elif c0 in ("follows","followersgained","newfollowers"):
            rename_map[c] = "ig_follows"
    df = df.rename(columns=rename_map)
    # Keep only date + known metrics
    keep = [date_col] + [c for c in ["ig_clicks","ig_reach","ig_impressions","ig_profile_visits","ig_follows"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={date_col: "date"})
    # Drop rows without a date
    df = df.dropna(subset=["date"])
    # Ensure daily resolution (group if dupes)
    agg = {c:"sum" for c in df.columns if c != "date"}
    df = df.groupby(df["date"].dt.date).agg(agg).reset_index().rename(columns={"index":"date"})
    df["date"] = pd.to_datetime(df["date"])
    # Fill missing metrics with 0 for later windows
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    return df

def load_daily_snapshot(base: Path) -> pd.DataFrame:
    path = base / "booking_outputs" / "daily_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the booking pipeline first to produce daily_snapshot.csv")
    d = pd.read_csv(path)
    # Required columns
    required = {"transaction_date"}
    if not required.issubset(set(c.lower() for c in d.columns)):
        # Try to standardize headers
        d.columns = [c.strip().lower() for c in d.columns]
    # Rename to canonical
    ren = {}
    if "transaction_date" not in d.columns and "txn_date" in d.columns:
        ren["txn_date"] = "transaction_date"
    if "bookings" not in d.columns and "num_bookings" in d.columns:
        ren["num_bookings"] = "bookings"
    if "guests" not in d.columns and "party_size" in d.columns:
        ren["party_size"] = "guests"
    if ren:
        d = d.rename(columns=ren)
    # Parse dates
    d["transaction_date"] = pd.to_datetime(d["transaction_date"], errors="coerce")
    # Coerce numeric
    for c in ["bookings","guests"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # We want aggregate new bookings/guests ADDED on each transaction_date across all future booking_dates
    daily = d.groupby(d["transaction_date"].dt.date).agg({"bookings":"sum","guests":"sum"}).reset_index()
    daily = daily.rename(columns={"transaction_date":"date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

def add_windows_and_lags(m: pd.DataFrame, metrics: list, lags=(1,2,3,4,5,6,7), windows=(7,)):
    m = m.sort_values("date").copy()
    for col in metrics:
        for L in lags:
            m[f"{col}_lag{L}"] = m[col].shift(L)
        for W in windows:
            m[f"{col}_{W}d"] = m[col].rolling(W, min_periods=1).sum()
    return m

def align_calendar(daily: pd.DataFrame, insta: pd.DataFrame) -> pd.DataFrame:
    # Outer join to keep full calendar
    cal = pd.DataFrame({"date": pd.date_range(min(daily["date"].min(), insta["date"].min()),
                                              max(daily["date"].max(), insta["date"].max()),
                                              freq="D")})
    out = cal.merge(daily, on="date", how="left").merge(insta, on="date", how="left")
    # Fill bookings/guests missing with 0
    for c in ["bookings","guests"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(float)
    # Fill IG metrics with 0 (assume silent days)
    ig_cols = [c for c in out.columns if c.startswith("ig_")]
    for c in ig_cols:
        out[c] = out[c].fillna(0).astype(float)
    return out

def compute_quick_corr(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("date",)]
    numeric = df[cols].select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame()
    corr = numeric.corr(method="pearson")
    # Keep only rows that include bookings/guests vs IG metrics
    focus_rows = [r for r in corr.index if r in ("bookings","guests")]
    focus_cols = [c for c in corr.columns if c.startswith("ig_")]
    if not focus_rows or not focus_cols:
        return corr
    return corr.loc[focus_rows, focus_cols].round(3)

def main(base_dir: Optional[str] = None):
    base = Path(base_dir) if base_dir else Path("/Users/zaklelex/Desktop/BM_data")
    base = base.expanduser().resolve()
    out_dir = base / "booking_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Base folder: {base}")
    daily = load_daily_snapshot(base)
    print(f"[i] Loaded daily_snapshot with {len(daily)} rows from {daily['date'].min().date()} to {daily['date'].max().date()}")

    insta_path = find_instagram_csv(base) or (base / "Insta_data.csv")
    if not insta_path.exists():
        raise FileNotFoundError(f"Could not find Instagram CSV under {base}. Put Insta_data.csv in the base folder or /marketing/")
    insta = parse_instagram_csv(insta_path)
    print(f"[i] Loaded Instagram data from {insta_path.name}: {len(insta)} days, {insta.columns.tolist()}")

    # build windows/lags on IG series
    ig_metrics = [c for c in insta.columns if c != "date"]
    insta_feat = add_windows_and_lags(insta, ig_metrics, lags=(1,2,3,4,5,6,7), windows=(7,))

    merged = align_calendar(daily, insta_feat)

    # Quick correlations
    corr = compute_quick_corr(merged)

    # Save
    merged_out = out_dir / "marketing_vs_bookings.csv"
    corr_out   = out_dir / "marketing_corr.csv"
    merged.to_csv(merged_out, index=False)
    corr.to_csv(corr_out)

    print(f"[✓] Wrote {merged_out}")
    print(f"[✓] Wrote {corr_out}")
    # Preview tail
    print(merged.tail(10).to_string(index=False))

if __name__ == "__main__":
    base_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(base_arg)

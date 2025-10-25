
#!/usr/bin/env python3
# booking_lead_etl_v2.py
# Build daily and cumulative snapshots from Tock Transactions, handling BOOKED/EDITED/RESCHEDULED/CANCELLED,
# and recording Booking Method. Produces:
#  - booking_outputs/daily_snapshot.csv
#  - booking_outputs/cumulative_asof_snapshot.csv
#  - booking_outputs/daily_snapshot_by_method.csv
#  - booking_outputs/cumulative_asof_by_method.csv
#
# Usage:
#   python3 booking_lead_etl_v2.py --in "/Users/zaklelex/Desktop/BM_data/the-brass-monkey-*.csv" --out "/Users/zaklelex/Desktop/BM_data/booking_outputs"
#   python3 booking_lead_etl_v2.py --in "/Users/zaklelex/Desktop/BM_data/the-brass-monkey-kansas-city-38420-Transactions-2025-09-01-2025-10-24.csv"
#
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV file or glob pattern (quotes recommended).")
    ap.add_argument("--out", dest="out", default=None, help="Output folder (default: ./booking_outputs next to input).")
    return ap.parse_args()

def _read_frames(pattern: str) -> pd.DataFrame:
    paths = sorted([str(p) for p in Path().glob(pattern)]) if not Path(pattern).exists() else [pattern]
    if not paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(2)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = p
            frames.append(df)
        except Exception as e:
            print(f"Failed reading {p}: {e}", file=sys.stderr)
    if not frames:
        print("No readable CSVs.", file=sys.stderr)
        sys.exit(2)
    return pd.concat(frames, ignore_index=True)

def _to_ts(dcol, tcol):
    d = pd.to_datetime(dcol, errors="coerce")
    t = pd.to_datetime(tcol, errors="coerce").dt.time
    # If time missing, set to midnight
    t = t.fillna(pd.Timestamp("00:00:00").time())
    return pd.to_datetime(d.dt.date.astype(str) + " " + pd.Series(t).astype(str), errors="coerce")

def compute_snapshots(raw: pd.DataFrame):
    # Normalize columns we need
    colmap = {
        "Transaction Date": "tx_date",
        "Transaction Time": "tx_time",
        "Action": "action",
        "Booking Date": "booking_date",
        "Booking Time": "booking_time",
        "Party Size": "party_size",
        "First Transaction ID": "first_tx_id",
        "Transaction ID": "tx_id",
        "Booking Method": "booking_method",
    }
    miss = [c for c in colmap if c not in raw.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = raw.rename(columns=colmap).copy()

    # Upper-case action, booking method clean
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["booking_method"] = df["booking_method"].fillna("Unknown").astype(str).str.strip()

    # Parse timestamps and dates
    df["tx_ts"] = _to_ts(df["tx_date"], df["tx_time"])
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce").dt.date
    df["booking_dt"] = pd.to_datetime(df["booking_date"], errors="coerce").dt.date

    # numeric party size
    df["party_size"] = pd.to_numeric(df["party_size"], errors="coerce").fillna(0).astype(int)

    # Keep only actions that affect future pace
    df = df[df["action"].isin(["BOOKED","EDITED","RESCHEDULED","CANCELLED"])].copy()

    # Order by reservation id then event time
    df = df.sort_values(["first_tx_id","tx_ts","tx_id"], kind="mergesort").reset_index(drop=True)

    # State machine per reservation
    # state[rid] = dict(party_size, booking_dt, method)
    state = {}
    led_rows = []  # ledger rows: (tx_date, booking_dt, guests_delta, bookings_delta, method)

    for _, row in df.iterrows():
        rid = row["first_tx_id"]
        act = row["action"]
        txd = row["tx_date"]
        book_d = row["booking_dt"]
        party = int(row["party_size"])
        method = row["booking_method"] or "Unknown"

        prev = state.get(rid)

        if act == "BOOKED":
            # New reservation
            led_rows.append((txd, book_d, party, 1, method))
            state[rid] = {"party_size": party, "booking_dt": book_d, "method": method}

        elif act in ("EDITED", "RESCHEDULED"):
            if prev is None:
                # Defensive: treat as a BOOKED if we never saw the start
                led_rows.append((txd, book_d, party, 1, method))
                state[rid] = {"party_size": party, "booking_dt": book_d, "method": method}
                continue

            prev_party = int(prev["party_size"])
            prev_date = prev["booking_dt"]
            prev_method = prev.get("method", method)

            if book_d != prev_date and pd.notna(book_d) and pd.notna(prev_date):
                # Move from prev_date -> book_d
                if prev_party != 0:
                    led_rows.append((txd, prev_date, -prev_party, 0, prev_method))
                if party != 0:
                    led_rows.append((txd, book_d, party, 0, prev_method))
            else:
                # Same date; party change only
                delta_party = party - prev_party
                if delta_party != 0:
                    led_rows.append((txd, book_d, delta_party, 0, prev_method))

            # Persist latest state; keep original method unless it changed to non-empty
            state[rid] = {"party_size": party, "booking_dt": book_d, "method": prev_method or method}

        elif act == "CANCELLED":
            if prev is not None:
                prev_party = int(prev["party_size"])
                prev_date = prev["booking_dt"]
                prev_method = prev.get("method", method)
                if prev_party != 0:
                    led_rows.append((txd, prev_date, -prev_party, -1, prev_method))
                # Remove from state (reservation no longer active)
                state.pop(rid, None)
            else:
                # No prior state -> nothing to subtract
                pass

    ledger = pd.DataFrame(led_rows, columns=["transaction_date","booking_date","guests","bookings","booking_method"])
    if ledger.empty:
        # Build empty outputs
        empty_cols = ["transaction_date","booking_date","guests","bookings"]
        return (pd.DataFrame(columns=empty_cols),
                pd.DataFrame(columns=empty_cols + ["as_of_date"]),
                pd.DataFrame(columns=empty_cols + ["booking_method"]),
                pd.DataFrame(columns=empty_cols + ["as_of_date","booking_method"]))

    # Aggregate duplicates
    daily = (ledger
             .groupby(["transaction_date","booking_date"], as_index=False)[["guests","bookings"]]
             .sum()
             .sort_values(["transaction_date","booking_date"]))

    daily_by_method = (ledger
                       .groupby(["transaction_date","booking_date","booking_method"], as_index=False)[["guests","bookings"]]
                       .sum()
                       .sort_values(["transaction_date","booking_date","booking_method"]))

    # Build cumulative as-of snapshot (overall)
    # For each booking_date, cumulative sum over transaction_date ascending
    tmp = daily.copy()
    tmp["as_of_date"] = tmp["transaction_date"]
    cum = (tmp.sort_values(["booking_date","as_of_date"])\
           .groupby("booking_date", group_keys=False)\
           .apply(lambda g: g.assign(cum_guests=g["guests"].cumsum(), cum_bookings=g["bookings"].cumsum()))\
           .loc[:, ["booking_date","as_of_date","guests","bookings","cum_guests","cum_bookings"]]\
           .reset_index(drop=True)[["booking_date","as_of_date","cum_guests","cum_bookings"]])

    # By method cumulative
    tmpm = daily_by_method.copy()
    tmpm["as_of_date"] = tmpm["transaction_date"]
    cumm = (tmpm.sort_values(["booking_date","booking_method","as_of_date"])\
            .groupby(["booking_date","booking_method"], group_keys=False)\
            .apply(lambda g: g.assign(cum_guests=g["guests"].cumsum(), cum_bookings=g["bookings"].cumsum()))\
            .loc[:, ["booking_date","booking_method","as_of_date","guests","bookings","cum_guests","cum_bookings"]]\
            .reset_index(drop=True)[["booking_date","booking_method","as_of_date","cum_guests","cum_bookings"]])

    # Ensure dtypes
    for c in ["guests","bookings","cum_guests","cum_bookings"]:
        if c in cum.columns: cum[c] = pd.to_numeric(cum[c], errors="coerce").fillna(0).astype(int)
        if c in cumm.columns: cumm[c] = pd.to_numeric(cumm[c], errors="coerce").fillna(0).astype(int)
        if c in daily.columns: daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0).astype(int)
        if c in daily_by_method.columns: daily_by_method[c] = pd.to_numeric(daily_by_method[c], errors="coerce").fillna(0).astype(int)

    return daily, cum, daily_by_method, cumm

def main():
    args = parse_args()
    inp = args.inp
    out = args.out

    if out is None:
        # default beside first input, in ./booking_outputs
        base = Path.cwd()
        out_dir = base / "booking_outputs"
    else:
        out_dir = Path(out)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_frames(inp)
    daily, cum, daily_by_method, cum_by_method = compute_snapshots(raw)

    # Write outputs
    daily.to_csv(out_dir / "daily_snapshot.csv", index=False)
    cum.to_csv(out_dir / "cumulative_asof_snapshot.csv", index=False)

    # Extended (by method)
    daily_by_method.to_csv(out_dir / "daily_snapshot_by_method.csv", index=False)
    cum_by_method.to_csv(out_dir / "cumulative_asof_by_method.csv", index=False)

    print(f"Wrote:")
    print(f"  {out_dir / 'daily_snapshot.csv'}")
    print(f"  {out_dir / 'cumulative_asof_snapshot.csv'}")
    print(f"  {out_dir / 'daily_snapshot_by_method.csv'}")
    print(f"  {out_dir / 'cumulative_asof_by_method.csv'}")

if __name__ == "__main__":
    main()

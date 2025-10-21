
# Generates a single self-contained HTML report with Plotly charts
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

def load_data(data_dir: Path):
    cum = pd.read_csv(data_dir / "cumulative_asof_snapshot.csv")
    cum["booking_date"] = pd.to_datetime(cum["booking_date"])
    cum["as_of_date"] = pd.to_datetime(cum["as_of_date"])
    if "cum_guests" not in cum.columns:
        raise ValueError("cum_guests missing")
    return cum

def build_report(data_dir: Path, out_html: Path):
    cum = load_data(data_dir)
    latest_as_of = cum["as_of_date"].max()

    # Choose next booking date and last week for sample charts
    upcoming = cum[cum["as_of_date"] == latest_as_of]["booking_date"].unique()
    if len(upcoming) == 0:
        upcoming_date = cum["booking_date"].max()
    else:
        upcoming_date = min(upcoming)
    last_week = upcoming_date - pd.Timedelta(days=7)

    # Curves
    this_curve = cum[cum["booking_date"] == upcoming_date].sort_values("as_of_date")
    last_curve = cum[cum["booking_date"] == last_week].sort_values("as_of_date")

    fig1 = px.line(this_curve, x="as_of_date", y="cum_guests", title=f"Cumulative guests for {upcoming_date.date()}")
    if not last_curve.empty:
        fig1.add_traces(px.line(last_curve, x="as_of_date", y="cum_guests").update_traces(name=f"{last_week.date()}").data)

    # Heatmap by lead time
    heat = cum.groupby(["booking_date", "lookahead_days"])["cum_guests"].mean().reset_index()
    pivot = heat.pivot(index="booking_date", columns="lookahead_days", values="cum_guests").fillna(0)
    fig2 = px.imshow(pivot, aspect="auto", origin="lower", color_continuous_scale="Blues", title="Density by lead time")

    # Save a single HTML with both
    html = "<h1>Booking Report</h1>" + pio.to_html(fig1, full_html=False, include_plotlyjs="cdn") + pio.to_html(fig2, full_html=False, include_plotlyjs=False)
    out_html.write_text(html, encoding="utf-8")

if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "booking_outputs"
    out = Path(__file__).resolve().parent / "booking_report.html"
    build_report(base, out)
    print(f"Wrote {out}")

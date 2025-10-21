
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Booking Lead-Time Visualizer", layout="wide")

# ------------------------------
# Locate data files
# ------------------------------
CANDIDATE_DIRS = [
    Path("/Users/zaklelex/Desktop/BM_data/booking_outputs"),
    Path.cwd() / "booking_outputs",
    Path(__file__).resolve().parent / "booking_outputs",
]

DATA_DIR = None
for p in CANDIDATE_DIRS:
    if (p / "cumulative_asof_snapshot.csv").exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    st.error("Place cumulative_asof_snapshot.csv and daily_snapshot.csv in a booking_outputs folder. Expected locations: ~/Desktop/BM_data/booking_outputs or ./booking_outputs")
    st.stop()

# ------------------------------
# Helpers
# ------------------------------
def fmt_date_opt(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            x = pd.to_datetime(x).date()
        if isinstance(x, datetime):
            x = x.date()
        if isinstance(x, date):
            return x.strftime("%Y-%m-%d")
        return str(x)
    except Exception:
        return str(x)

def to_py_dates(series):
    return (
        pd.to_datetime(series, errors="coerce")
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

# ------------------------------
# Load data
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir: Path):
    cum = pd.read_csv(data_dir / "cumulative_asof_snapshot.csv")
    daily = pd.read_csv(data_dir / "daily_snapshot.csv")
    wow_path = data_dir / "week_over_week_latest.csv"
    wow = pd.read_csv(wow_path) if wow_path.exists() else pd.DataFrame()

    # Parse dates
    for col in ["booking_date", "as_of_date"]:
        if col in cum.columns:
            cum[col] = pd.to_datetime(cum[col], errors="coerce")
    for col in ["transaction_date", "booking_date"]:
        if col in daily.columns:
            daily[col] = pd.to_datetime(daily[col], errors="coerce")

    # Safe metric columns
    if "cum_guests" not in cum.columns:
        cum["cum_guests"] = np.nan
    if "cum_bookings" not in cum.columns:
        cum["cum_bookings"] = np.nan
    if "guests" not in daily.columns:
        daily["guests"] = np.nan
    if "bookings" not in daily.columns:
        daily["bookings"] = np.nan

    # Derive daily adds from cumulative by booking_date
    cum_sorted = cum.sort_values(["booking_date", "as_of_date"]).copy()
    cum_sorted["daily_add_guests"] = cum_sorted.groupby("booking_date")["cum_guests"].diff().fillna(cum_sorted["cum_guests"])
    cum_sorted["daily_add_bookings"] = cum_sorted.groupby("booking_date")["cum_bookings"].diff().fillna(cum_sorted["cum_bookings"])

    # Add helpers
    cum_sorted["booking_dow"] = cum_sorted["booking_date"].dt.day_name()
    cum_sorted["week"] = cum_sorted["booking_date"].dt.isocalendar().week.astype('Int64')

    return cum_sorted, daily, wow

cum, daily, wow = load_data(DATA_DIR)

# Derived sets
all_booking_dates = to_py_dates(cum["booking_date"])
if len(all_booking_dates) == 0:
    st.warning("No booking dates found in cumulative_asof_snapshot.csv")
    st.stop()

latest_as_of_ts = pd.to_datetime(cum["as_of_date"]).max()
latest_as_of = latest_as_of_ts.date() if pd.notna(latest_as_of_ts) else date.today()

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Controls")

metric = st.sidebar.radio("Metric", ["Guests", "Bookings"], index=0)
view = st.sidebar.radio("View", ["Cumulative", "Daily adds"], index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Compare two dates", "Compare day of week", "Upcoming pacing"],
    index=0
)

def metric_cols(prefix=""):
    if metric == "Guests":
        return f"{prefix}cum_guests", f"{prefix}daily_add_guests"
    return f"{prefix}cum_bookings", f"{prefix}daily_add_bookings"

# ------------------------------
# Helper functions
# ------------------------------
def curve_for_dates(dates, align="lead"):
    m_cum, m_add = metric_cols()
    out = []
    for d in dates:
        d_ts = pd.to_datetime(d)
        sub = cum[cum["booking_date"] == d_ts].copy()
        if sub.empty:
            continue
        y = m_cum if view == "Cumulative" else m_add
        x = "lookahead_days" if align == "lead" else "as_of_date"
        if y not in sub.columns or x not in sub.columns:
            continue
        sub = sub.sort_values(x)
        sub["series"] = fmt_date_opt(d)
        out.append(sub[[x, y, "series"]].rename(columns={x: "x", y: "y"}))
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["x", "y", "series"])

def dow_profile(day_name):
    """Average curve for a given day of week by lookahead_days."""
    m_cum, m_add = metric_cols()
    ycol = m_cum if view == "Cumulative" else m_add
    tmp = cum[cum["booking_dow"] == day_name]
    prof = tmp.groupby("lookahead_days")[ycol].mean().reset_index(name="y")
    prof["series"] = f"Avg {day_name}"
    prof = prof.rename(columns={"lookahead_days": "x"})
    return prof

def dow_percentile_band(day_name, q_low=0.2, q_high=0.8):
    """Percentile band for visualization."""
    m_cum, m_add = metric_cols()
    ycol = m_cum if view == "Cumulative" else m_add
    tmp = cum[cum["booking_dow"] == day_name]
    band = tmp.groupby("lookahead_days")[ycol].quantile([q_low, 0.5, q_high]).unstack().reset_index()
    band.columns = ["x", "p_low", "median", "p_high"]
    return band

def pacing_table(today=None, horizon=21):
    if today is None:
        today = latest_as_of
    m_cum, _ = metric_cols()
    # Candidate upcoming dates
    future_dates = [d for d in all_booking_dates if d >= today and (d - today).days <= horizon]
    rows = []
    for bd in future_dates:
        # Current cumulative as of today
        cur = cum[(cum["booking_date"] == pd.to_datetime(bd)) & (cum["as_of_date"] == pd.to_datetime(today))]
        cur_val = cur[m_cum].iloc[0] if len(cur) else np.nan

        # Same lead a week ago
        lead = (pd.to_datetime(bd) - pd.to_datetime(today)).days
        last_week_asof = pd.to_datetime(today) - pd.Timedelta(days=7)
        last_week_bd = pd.to_datetime(bd) - pd.Timedelta(days=7)
        prev = cum[(cum["booking_date"] == last_week_bd) & (cum["as_of_date"] == last_week_asof)]
        prev_val = prev[m_cum].iloc[0] if len(prev) else np.nan

        # Day of week average at this lookahead
        dow = pd.to_datetime(bd).day_name()
        avg = cum[cum["booking_dow"] == dow].groupby("lookahead_days")[m_cum].mean()
        avg_val = avg.get(lead, np.nan)

        # Pace index vs average
        pace = cur_val / avg_val if pd.notna(cur_val) and pd.notna(avg_val) and avg_val != 0 else np.nan

        rows.append({
            "booking_date": bd,
            "lookahead_days": int(lead),
            "current": float(cur_val) if pd.notna(cur_val) else np.nan,
            "last_week_same_lead": float(prev_val) if pd.notna(prev_val) else np.nan,
            "dow_avg_same_lead": float(avg_val) if pd.notna(avg_val) else np.nan,
            "pace_index": float(pace) if pd.notna(pace) else np.nan
        })
    return pd.DataFrame(rows)

# ------------------------------
# UI by mode
# ------------------------------
if mode == "Compare two dates":
    idx1 = len(all_booking_dates) - 1
    idx2 = max(idx1 - 7, 0)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        d1 = st.selectbox("Date A", all_booking_dates, index=idx1, format_func=fmt_date_opt)
    with col2:
        d2 = st.selectbox("Date B", all_booking_dates, index=idx2, format_func=fmt_date_opt)
    with col3:
        align = st.radio("Align by", ["Lead time", "Calendar date"], index=0)

    align_key = "lead" if align == "Lead time" else "calendar"
    data = curve_for_dates([d1, d2], align=align_key)
    title_metric = metric.lower()
    title_view = "cumulative" if view == "Cumulative" else "daily adds"
    title_align = "lead time" if align_key == "lead" else "calendar date"
    st.subheader(f"Compare {title_metric}, {title_view}, aligned by {title_align}")
    if data.empty:
        st.info("No data for the selected dates.")
    else:
        fig = px.line(data, x="x", y="y", color="series", markers=True)
        fig.update_layout(legend_title="Booking date", xaxis_title=("Lookahead days" if align_key=="lead" else "As of date"), yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    m_cum, m_add = metric_cols()
    for i, d, col in [(1, d1, k1), (2, d2, k2)]:
        final = cum[cum["booking_date"] == pd.to_datetime(d)]
        if not final.empty and m_cum in final.columns:
            final_val = final.sort_values("as_of_date")[m_cum].iloc[-1]
            adds_24h = final.sort_values("as_of_date")[m_add].iloc[-1] if m_add in final.columns else np.nan
        else:
            final_val, adds_24h = np.nan, np.nan
        col.metric(f"{fmt_date_opt(d)} total", int(final_val) if pd.notna(final_val) else 0)
        col.metric(f"{fmt_date_opt(d)} last add", int(adds_24h) if pd.notna(adds_24h) else 0)

elif mode == "Compare day of week":
    dows = sorted([d for d in cum["booking_dow"].dropna().unique().tolist()])
    if len(dows) == 0:
        st.info("No day of week values in data.")
        st.stop()
    dow = st.selectbox("Day of week", dows, index=0)

    # Select a reference date with that DOW, default to most recent
    choices = to_py_dates(cum.loc[cum["booking_date"].dt.day_name() == dow, "booking_date"])
    if len(choices) == 0:
        st.info("No dates for that day of week.")
        st.stop()
    ref = st.selectbox("Reference booking date", choices, index=len(choices)-1, format_func=fmt_date_opt)

    prof = dow_profile(dow)
    band = dow_percentile_band(dow)

    ref_curve = curve_for_dates([ref], align="lead")
    ref_curve["series"] = f"{fmt_date_opt(ref)}"
    prof["series"] = f"Avg {dow}"

    title_metric = metric.lower()
    title_view = "cumulative" if view == "Cumulative" else "daily adds"

    st.subheader(f"{dow} profile vs reference, {title_metric}, {title_view}")
    fig = px.line(ref_curve, x="x", y="y", color="series", markers=True)
    if not prof.empty:
        fig.add_traces(px.line(prof, x="x", y="y", color="series").data)
    if not band.empty:
        fig.add_traces(px.area(band, x="x", y="p_high").update_traces(name="80th pct").data)
        fig.add_traces(px.area(band, x="x", y="p_low").update_traces(name="20th pct").data)
    fig.update_layout(xaxis_title="Lookahead days", yaxis_title=metric, legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap for this DOW
    st.subheader(f"{dow} density by lead time")
    m_cum, m_add = metric_cols()
    ycol = m_cum if view == "Cumulative" else m_add
    heat = cum[cum["booking_dow"] == dow].groupby(["booking_date", "lookahead_days"])[ycol].mean().reset_index()
    if heat.empty:
        st.info("No data for this day of week.")
    else:
        heat_pivot = heat.pivot(index="booking_date", columns="lookahead_days", values=ycol).fillna(0)
        fig2 = px.imshow(heat_pivot, aspect="auto", origin="lower", color_continuous_scale="Blues")
        fig2.update_layout(xaxis_title="Lookahead days", yaxis_title="Booking date")
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Upcoming pacing
    st.subheader("Upcoming pacing table")
    horizon = st.slider("Horizon days", 7, 42, 21)
    today = st.date_input("As of date", latest_as_of)
    pace = pacing_table(today=today, horizon=horizon)
    if pace.empty:
        st.info("No upcoming dates in range.")
    else:
        # Display table
        show = pace.copy()
        show["current"] = show["current"].round(0).astype("Int64")
        show["last_week_same_lead"] = show["last_week_same_lead"].round(0).astype("Int64")
        show["dow_avg_same_lead"] = show["dow_avg_same_lead"].round(1)
        show["pace_index"] = show["pace_index"].round(2)
        st.dataframe(show, use_container_width=True)

        # Bar of delta vs last week
        st.subheader("Delta vs last week at same lead")
        tmp = pace.copy()
        tmp["delta_vs_last_week"] = tmp["current"] - tmp["last_week_same_lead"]
        tmp = tmp.dropna(subset=["delta_vs_last_week"])
        if not tmp.empty:
            fig3 = px.bar(tmp, x="booking_date", y="delta_vs_last_week")
            fig3.update_layout(xaxis_title="Booking date", yaxis_title=f"Delta {metric}")
            st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            "Download pacing table CSV",
            data=pace.to_csv(index=False).encode("utf-8"),
            file_name=f"pacing_{today}.csv",
            mime="text/csv"
        )

st.caption(f"Data folder: {DATA_DIR}")


# visualizer3.py
import os
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Booking Lead-Time Visualizer", layout="wide")

# ------------------------------
# Config
# ------------------------------
RAW_BASE = st.secrets.get("RAW_BASE", "").rstrip("/")

LOCAL_DIRS = [
    Path.cwd() / "booking_outputs",
    Path(__file__).resolve().parent / "booking_outputs",
    Path.home() / "Desktop" / "BM_data" / "booking_outputs",
]

REQUIRED_FILES = ["cumulative_asof_snapshot.csv", "daily_snapshot.csv"]

# ------------------------------
# Data loading helpers
# ------------------------------
def try_local_dirs():
    for d in LOCAL_DIRS:
        if all((d / f).exists() for f in REQUIRED_FILES):
            return d
    return None

def read_from_raw_base():
    if not RAW_BASE:
        return None
    try:
        cum = pd.read_csv(f"{RAW_BASE}/cumulative_asof_snapshot.csv")
        daily = pd.read_csv(f"{RAW_BASE}/daily_snapshot.csv")
        try:
            wow = pd.read_csv(f"{RAW_BASE}/week_over_week_latest.csv")
        except Exception:
            wow = pd.DataFrame()
        return cum, daily, wow
    except Exception:
        return None

def upload_ui():
    st.info("Upload two files to proceed.")
    up_cum = st.file_uploader("Upload cumulative_asof_snapshot.csv", type=["csv"], key="cum")
    up_daily = st.file_uploader("Upload daily_snapshot.csv", type=["csv"], key="daily")
    if up_cum is None or up_daily is None:
        st.stop()
    cum = pd.read_csv(up_cum)
    daily = pd.read_csv(up_daily)
    wow = pd.DataFrame()
    return cum, daily, wow

@st.cache_data(ttl=300, show_spinner=False)
def load_core_data():
    # 1) Local files in repo
    local_dir = try_local_dirs()
    if local_dir:
        cum = pd.read_csv(local_dir / "cumulative_asof_snapshot.csv")
        daily = pd.read_csv(local_dir / "daily_snapshot.csv")
        wow_path = local_dir / "week_over_week_latest.csv"
        wow = pd.read_csv(wow_path) if wow_path.exists() else pd.DataFrame()
        source = f"Local: {local_dir}"
        return cum, daily, wow, source

    # 2) GitHub raw (via secret RAW_BASE)
    raw = read_from_raw_base()
    if raw:
        cum, daily, wow = raw
        source = f"GitHub raw: {RAW_BASE}"
        return cum, daily, wow, source

    # 3) Upload UI
    cum, daily, wow = upload_ui()
    source = "Uploaded"
    return cum, daily, wow, source

def try_load_marketing_from_local():
    for d in LOCAL_DIRS:
        mvb = d / "marketing_vs_bookings.csv"
        mco = d / "marketing_corr.csv"
        if mvb.exists():
            mvb_df = pd.read_csv(mvb)
            mco_df = pd.read_csv(mco) if mco.exists() else pd.DataFrame()
            return mvb_df, mco_df, f"Local: {d}"
    return None

def try_load_marketing_from_raw():
    if not RAW_BASE:
        return None
    try:
        mvb = pd.read_csv(f"{RAW_BASE}/marketing_vs_bookings.csv")
        try:
            mco = pd.read_csv(f"{RAW_BASE}/marketing_corr.csv")
        except Exception:
            mco = pd.DataFrame()
        return mvb, mco, f"GitHub raw: {RAW_BASE}"
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_marketing():
    local = try_load_marketing_from_local()
    if local:
        return local
    raw = try_load_marketing_from_raw()
    if raw:
        return raw
    # No marketing files
    return None, None, None

# ------------------------------
# Small utilities
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

def num(x, zero="0"):
    try:
        return f"{int(x):,}"
    except Exception:
        return zero

def zscore(col: pd.Series) -> pd.Series:
    col = pd.to_numeric(col, errors="coerce")
    mu = col.mean()
    sd = col.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return col * 0
    return (col - mu) / sd

# ------------------------------
# Transform core data
# ------------------------------
cum_raw, daily_raw, wow_raw, data_source = load_core_data()

# Parse dates
for col in ["booking_date", "as_of_date"]:
    if col in cum_raw.columns:
        cum_raw[col] = pd.to_datetime(cum_raw[col], errors="coerce")
for col in ["transaction_date", "booking_date"]:
    if col in daily_raw.columns:
        daily_raw[col] = pd.to_datetime(daily_raw[col], errors="coerce")

# Safe metric columns
for col in ["cum_guests", "cum_bookings"]:
    if col not in cum_raw.columns:
        cum_raw[col] = np.nan
for col in ["guests", "bookings"]:
    if col not in daily_raw.columns:
        daily_raw[col] = np.nan

cum = cum_raw.sort_values(["booking_date", "as_of_date"]).copy()
cum["daily_add_guests"]   = cum.groupby("booking_date")["cum_guests"].diff().fillna(cum["cum_guests"])
cum["daily_add_bookings"] = cum.groupby("booking_date")["cum_bookings"].diff().fillna(cum["cum_bookings"])

cum["booking_dow"] = cum["booking_date"].dt.day_name()
cum["lookahead_days"] = cum["lookahead_days"].astype("int32", errors="ignore")
all_booking_dates = to_py_dates(cum["booking_date"])
latest_as_of_ts = pd.to_datetime(cum["as_of_date"]).max()
latest_as_of = latest_as_of_ts.date() if pd.notna(latest_as_of_ts) else date.today()

# ------------------------------
# Sidebar and state
# ------------------------------
st.sidebar.header("Controls")
metric = st.sidebar.radio("Metric", ["Guests", "Bookings"], index=0)
view   = st.sidebar.radio("View", ["Cumulative", "Daily adds"], index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Overview", "Compare dates", "Compare day of week", "Upcoming pacing", "Marketing vs bookings"],
    index=0
)

def metric_cols(prefix=""):
    if metric == "Guests":
        return f"{prefix}cum_guests", f"{prefix}daily_add_guests"
    return f"{prefix}cum_bookings", f"{prefix}daily_add_bookings"

# ------------------------------
# Reusable builders
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

def dow_profile(day_name, q_low=0.1, q_high=0.9):
    m_cum, m_add = metric_cols()
    ycol = m_cum if view == "Cumulative" else m_add
    tmp = cum[cum["booking_dow"] == day_name]
    prof = tmp.groupby("lookahead_days")[ycol].median().reset_index(name="median")
    band = tmp.groupby("lookahead_days")[ycol].quantile([q_low, q_high]).unstack().reset_index()
    band.columns = ["lookahead_days", "p_low", "p_high"]
    return prof.rename(columns={"lookahead_days": "x"}), band.rename(columns={"lookahead_days": "x"})

def pacing_table(today=None, horizon=21):
    if today is None:
        today = latest_as_of
    m_cum, _ = metric_cols()
    future_dates = [d for d in all_booking_dates if d >= today and (d - today).days <= horizon]
    rows = []
    for bd in future_dates:
        cur = cum[(cum["booking_date"] == pd.to_datetime(bd)) & (cum["as_of_date"] == pd.to_datetime(today))]
        cur_val = cur[m_cum].iloc[0] if len(cur) else np.nan
        lead = (pd.to_datetime(bd) - pd.to_datetime(today)).days
        last_week_asof = pd.to_datetime(today) - pd.Timedelta(days=7)
        last_week_bd = pd.to_datetime(bd) - pd.Timedelta(days=7)
        prev = cum[(cum["booking_date"] == last_week_bd) & (cum["as_of_date"] == last_week_asof)]
        prev_val = prev[m_cum].iloc[0] if len(prev) else np.nan
        dow = pd.to_datetime(bd).day_name()
        avg = cum[cum["booking_dow"] == dow].groupby("lookahead_days")[m_cum].mean()
        avg_val = avg.get(lead, np.nan)
        pace = cur_val / avg_val if pd.notna(cur_val) and pd.notna(avg_val) and avg_val != 0 else np.nan
        rows.append({
            "booking_date": bd,
            "lookahead_days": int(lead),
            "current": float(cur_val) if pd.notna(cur_val) else np.nan,
            "last_week_same_lead": float(prev_val) if pd.notna(prev_val) else np.nan,
            "dow_avg_same_lead": float(avg_val) if pd.notna(avg_val) else np.nan,
            "pace_index": float(pace) if pd.notna(pace) else np.nan
        })
    df = pd.DataFrame(rows)
    return df

# ------------------------------
# Top KPIs
# ------------------------------
st.caption(f"Data source: {data_source}")
k1, k2, k3, k4 = st.columns(4)
m_cum, m_add = metric_cols()

if len(all_booking_dates) >= 1:
    last_date = all_booking_dates[-1]
    last_curve = cum[cum["booking_date"] == pd.to_datetime(last_date)]
    last_total = last_curve[m_cum].max() if not last_curve.empty else 0
    last_add = last_curve[m_add].iloc[-1] if not last_curve.empty and m_add in last_curve.columns else 0
else:
    last_date, last_total, last_add = None, 0, 0

def next_weekend_totals(as_of):
    ref = pd.to_datetime(as_of)
    fri = (ref + pd.Timedelta(days=(4 - ref.weekday()) % 7)).date()
    sat = (ref + pd.Timedelta(days=(5 - ref.weekday()) % 7)).date()
    out = {}
    for d in [fri, sat]:
        row = cum[(cum["booking_date"] == pd.to_datetime(d)) & (cum["as_of_date"] == pd.to_datetime(as_of))]
        out[d] = row[m_cum].iloc[0] if len(row) else np.nan
    return fri, out.get(fri), sat, out.get(sat)

fri, fri_val, sat, sat_val = next_weekend_totals(latest_as_of)

k1.metric("As of", fmt_date_opt(latest_as_of))
k2.metric(f"Last date total", num(last_total))
k3.metric(f"Next Fri {fmt_date_opt(fri)}", num(fri_val))
k4.metric(f"Next Sat {fmt_date_opt(sat)}", num(sat_val))

st.divider()

# ------------------------------
# Modes
# ------------------------------
if mode == "Overview":
    st.subheader("Lead-time density by booking date")
    ycol = m_cum if view == "Cumulative" else m_add
    heat = cum.groupby(["booking_date", "lookahead_days"])[ycol].mean().reset_index()
    if heat.empty:
        st.info("No data in current selection.")
    else:
        heat_pivot = heat.pivot(index="booking_date", columns="lookahead_days", values=ycol).fillna(0)
        fig = px.imshow(heat_pivot, aspect="auto", origin="lower", color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="Lookahead days", yaxis_title="Booking date")
        st.plotly_chart(fig, use_container_width=True)

elif mode == "Compare dates":
    st.write("Pick up to four dates.")
    default = all_booking_dates[-2:] if len(all_booking_dates) >= 2 else all_booking_dates
    dates = st.multiselect("Dates", all_booking_dates, default=default, format_func=fmt_date_opt, max_selections=4)
    align = st.radio("Align by", ["Lead time", "Calendar date"], index=0, horizontal=True)
    align_key = "lead" if align == "Lead time" else "calendar"
    data = curve_for_dates(dates, align=align_key)
    if data.empty:
        st.info("Select dates with curves available.")
    else:
        fig = px.line(data, x="x", y="y", color="series", markers=True)
        fig.update_layout(legend_title="Booking date", xaxis_title=("Lookahead days" if align_key=="lead" else "As of date"), yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

elif mode == "Compare day of week":
    dows = sorted([d for d in cum["booking_dow"].dropna().unique().tolist()])
    dow = st.selectbox("Day of week", dows, index=0)
    choices = to_py_dates(cum.loc[cum["booking_date"].dt.day_name() == dow, "booking_date"])
    ref = st.selectbox("Reference booking date", choices, index=len(choices)-1 if choices else 0, format_func=fmt_date_opt)
    prof, band = dow_profile(dow)
    ref_curve = curve_for_dates([ref], align="lead")

    st.subheader(f"{dow} profile vs reference")
    fig = px.line(ref_curve, x="x", y="y", color="series", markers=True)
    if not prof.empty:
        fig.add_traces(px.line(prof, x="x", y="median").update_traces(name=f"Median {dow}").data)
    if not band.empty:
        fig.add_traces(px.area(band, x="x", y="p_high").update_traces(name="90th pct").data)
        fig.add_traces(px.area(band, x="x", y="p_low").update_traces(name="10th pct").data)
    fig.update_layout(xaxis_title="Lookahead days", yaxis_title=metric, legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Upcoming pacing":
    st.subheader("Upcoming pacing")
    horizon = st.slider("Horizon days", 7, 42, 21)
    today = st.date_input("As of date", latest_as_of)
    pace = pacing_table(today=today, horizon=horizon)
    if pace.empty:
        st.info("No upcoming dates in range.")
    else:
        show = pace.copy()
        # Targets support: optional booking_outputs/targets.csv
        targets_path = None
        for d in LOCAL_DIRS:
            if (d / "targets.csv").exists():
                targets_path = d / "targets.csv"
                break
        if targets_path:
            targets = pd.read_csv(targets_path, parse_dates=["booking_date"])
            targets["booking_date"] = targets["booking_date"].dt.date
            show = show.merge(targets, on="booking_date", how="left")
            show["to_target"] = (show["target_guests"] - show["current"]).round(0)

        show["pace_index"] = show["pace_index"].round(2)
        show["current"] = show["current"].round(0).astype("Int64")
        show["last_week_same_lead"] = show["last_week_same_lead"].round(0).astype("Int64")
        show["dow_avg_same_lead"] = show["dow_avg_same_lead"].round(1)

        st.dataframe(show, use_container_width=True)

        st.subheader("Delta vs last week at same lead")
        tmp = show.copy()
        tmp["delta_vs_last_week"] = tmp["current"] - tmp["last_week_same_lead"]
        tmp = tmp.dropna(subset=["delta_vs_last_week"])
        if not tmp.empty:
            fig3 = px.bar(tmp, x="booking_date", y="delta_vs_last_week")
            fig3.update_layout(xaxis_title="Booking date", yaxis_title=f"Delta {metric}")
            st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            "Download pacing table CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"pacing_{today}.csv",
            mime="text/csv"
        )

else:  # Marketing vs bookings
    st.subheader("Marketing vs bookings (daily, by transaction date)")
    mvb_df, mco_df, m_source = load_marketing()

    if mvb_df is None or mvb_df.empty:
        st.info(
            "No marketing files found. Expected **marketing_vs_bookings.csv** (and optional **marketing_corr.csv**) "
            "under one of these folders: "
            f"{[str(p) for p in LOCAL_DIRS]}. "
            "Generate them with booking_lead_analysis2.py, then redeploy."
        )
    else:
        # Parse and prep
        mvb = mvb_df.copy()
        if "date" not in mvb.columns:
            st.error("marketing_vs_bookings.csv must include a 'date' column.")
            st.stop()
        mvb["date"] = pd.to_datetime(mvb["date"], errors="coerce")
        mvb = mvb.sort_values("date")
        # figure IG columns
        ig_cols = [c for c in mvb.columns if c.startswith("ig_")]
        core_cols = ["bookings", "guests"]
        # Controls
        min_d, max_d = mvb["date"].min().date(), mvb["date"].max().date()
        start, end = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        # sanitize range
        if isinstance(start, (list, tuple)):
            start, end = start
        if start > end:
            start, end = end, start

        mvb_slice = mvb[(mvb["date"].dt.date >= start) & (mvb["date"].dt.date <= end)].copy()

        left_metric = st.radio("Bookings axis", core_cols, index=0, horizontal=True)
        default_igs = [c for c in ["ig_reach_7d","ig_clicks_7d","ig_impressions_7d"] if c in ig_cols][:2] or ig_cols[:2]
        picked_igs = st.multiselect("Pick IG series", ig_cols, default=default_igs)
        normalize = st.checkbox("Normalize series (z-score) on one axis", value=True)

        if len(picked_igs) == 0:
            st.info("Select at least one IG series.")
        else:
            if normalize:
                plot_df = mvb_slice[["date", left_metric] + picked_igs].copy()
                for c in [left_metric] + picked_igs:
                    plot_df[c] = zscore(plot_df[c])
                long = plot_df.melt(id_vars="date", value_vars=[left_metric] + picked_igs, var_name="series", value_name="z")
                fig = px.line(long, x="date", y="z", color="series")
                fig.update_layout(yaxis_title="Z-score (normalized)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Dual-axis if single IG; else one-axis normalized fallback above is suggested
                if len(picked_igs) == 1:
                    y2 = picked_igs[0]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=mvb_slice["date"], y=mvb_slice[left_metric], name=left_metric), secondary_y=False)
                    fig.add_trace(go.Scatter(x=mvb_slice["date"], y=mvb_slice[y2], name=y2), secondary_y=True)
                    fig.update_yaxes(title_text=left_metric, secondary_y=False)
                    fig.update_yaxes(title_text=y2, secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("For multiple IG series, enable normalization to compare on one axis.")

            st.caption(f"Marketing source: {m_source}")

            st.subheader("Correlations (Pearson)")
            if mco_df is not None and not mco_df.empty:
                st.dataframe(mco_df, use_container_width=True)
            else:
                # compute on the fly for the selected window
                corr_cols = [left_metric] + picked_igs
                sub = mvb_slice[corr_cols].select_dtypes(include=[np.number]).copy()
                if sub.shape[0] >= 3:
                    corr = sub.corr().round(3)
                    st.dataframe(corr, use_container_width=True)
                else:
                    st.info("Not enough rows in the selected date range to compute correlation.")

            st.download_button(
                "Download filtered marketing_vs_bookings.csv",
                data=mvb_slice.to_csv(index=False).encode("utf-8"),
                file_name=f"marketing_vs_bookings_{start}_to_{end}.csv",
                mime="text/csv"
            )

# Utility row
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.experimental_rerun()
with colB:
    st.caption(f"Data source: {data_source}")
with colC:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

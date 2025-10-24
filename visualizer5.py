# visualizer5.py
import os
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Bookings & Marketing Overview", layout="wide")

# ==============================
# Config
# ==============================
RAW_BASE = st.secrets.get("RAW_BASE", "").rstrip("/")

LOCAL_DIRS = [
    Path.cwd() / "booking_outputs",
    Path(__file__).resolve().parent / "booking_outputs",
    Path.home() / "Desktop" / "BM_data" / "booking_outputs",
]

REQUIRED_FILES = ["cumulative_asof_snapshot.csv", "daily_snapshot.csv"]

# ==============================
# Local timezone + 'today' helper
# ==============================
try:
    from zoneinfo import ZoneInfo  # py3.9+
    CT = ZoneInfo("America/Chicago")
except Exception:
    CT = None

def today_local():
    return (datetime.now(CT).date() if CT else datetime.now().date())

# ==============================
# Query params (deep-link saved views)
# ==============================
def _get_params():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_params(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

qs = _get_params()

# ==============================
# Loaders
# ==============================
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
    local_dir = try_local_dirs()
    if local_dir:
        cum = pd.read_csv(local_dir / "cumulative_asof_snapshot.csv")
        daily = pd.read_csv(local_dir / "daily_snapshot.csv")
        wow_path = local_dir / "week_over_week_latest.csv"
        wow = pd.read_csv(wow_path) if wow_path.exists() else pd.DataFrame()
        source = f"Local: {local_dir}"
        return cum, daily, wow, source

    raw = read_from_raw_base()
    if raw:
        cum, daily, wow = raw
        source = f"GitHub raw: {RAW_BASE}"
        return cum, daily, wow, source

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
    return None, None, None

# Optional campaigns overlay
def _load_campaigns():
    for d in LOCAL_DIRS:
        p = d / "campaigns.csv"
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=["start_date","end_date"])
            except Exception:
                pass
    return pd.DataFrame()

# ==============================
# Utilities
# ==============================
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

def zscore(col: pd.Series) -> pd.Series:
    col = pd.to_numeric(col, errors="coerce")
    mu = col.mean()
    sd = col.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return col * 0
    return (col - mu) / sd

def num(x, zero="0"):
    try:
        return f"{int(x):,}"
    except Exception:
        try:
            return f"{float(x):,.1f}"
        except Exception:
            return zero

# ==============================
# Transform core data
# ==============================
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

# Build cum and daily adds
cum = cum_raw.sort_values(["booking_date", "as_of_date"]).copy()
cum["daily_add_guests"]   = cum.groupby("booking_date")["cum_guests"].diff().fillna(cum["cum_guests"])
cum["daily_add_bookings"] = cum.groupby("booking_date")["cum_bookings"].diff().fillna(cum["cum_bookings"])

cum["booking_dow"] = cum["booking_date"].dt.day_name()
cum["lookahead_days"] = cum["lookahead_days"].astype("int32", errors="ignore")

# ---- Clamp by 'today' in America/Chicago so future as_of rows never appear
cum["as_of_date"] = pd.to_datetime(cum["as_of_date"], errors="coerce")
TODAY = today_local()
cum = cum[cum["as_of_date"].dt.date <= TODAY].copy()

# Recompute latest_as_of after clamping
latest_as_of_ts = pd.to_datetime(cum["as_of_date"]).max()
latest_as_of = (latest_as_of_ts.date() if pd.notna(latest_as_of_ts) else TODAY)

all_booking_dates = to_py_dates(cum["booking_date"])

# Weekday baseline (median by lead time)
def weekday_baseline(metric_key: str):
    base = cum.groupby(["booking_dow", "lookahead_days"])[metric_key].median().reset_index()
    return base

# ==============================
# Sidebar
# ==============================
_mode_opts = ["Executive overview","Compare dates","Compare day of week","Upcoming pacing","Marketing vs bookings"]
_mode_qs = (qs.get("mode") or ["Executive_overview"])[0].replace("_"," ")
_mode_idx = _mode_opts.index(_mode_qs) if _mode_qs in _mode_opts else 0

_metric_qs = (qs.get("metric") or ["Guests"])[0]
_metric_idx = 0 if _metric_qs=="Guests" else 1

_view_qs = (qs.get("view") or ["Cumulative"])[0].replace("_"," ")
_view_idx = 0 if _view_qs=="Cumulative" else 1

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", _mode_opts, index=_mode_idx)
metric = st.sidebar.radio("Metric", ["Guests", "Bookings"], index=_metric_idx)
view   = st.sidebar.radio("View", ["Cumulative", "Daily adds"], index=_view_idx)

# Keep URL in sync so the view is shareable
_set_params(
    mode=mode.replace(" ","_"),
    metric=metric,
    view=view.replace(" ","_")
)

def metric_cols(prefix=""):
    if metric == "Guests":
        return f"{prefix}cum_guests", f"{prefix}daily_add_guests"
    return f"{prefix}cum_bookings", f"{prefix}daily_add_bookings"

# ==============================
# Reusable builders
# ==============================
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
        avg = cum[cum["booking_dow"] == dow].groupby("lookahead_days")[m_cum].median()
        avg_val = avg.get(lead, np.nan)
        pace = cur_val / avg_val if pd.notna(cur_val) and pd.notna(avg_val) and avg_val != 0 else np.nan
        rows.append({
            "booking_date": bd,
            "lookahead_days": int(lead),
            "current": float(cur_val) if pd.notna(cur_val) else np.nan,
            "last_week_same_lead": float(prev_val) if pd.notna(prev_val) else np.nan,
            "dow_median_same_lead": float(avg_val) if pd.notna(avg_val) else np.nan,
            "pace_index": float(pace) if pd.notna(pace) else np.nan
        })
    df = pd.DataFrame(rows)
    return df

def next_weekday(dts, weekday_int):
    # 0=Mon ... 4=Fri 5=Sat 6=Sun
    days_ahead = (weekday_int - dts.weekday() + 7) % 7
    if days_ahead == 0:  # "next" never means "today"
        days_ahead = 7
    return (dts + pd.Timedelta(days=days_ahead)).date()

# ==============================
# Executive Overview
# ==============================
if mode == "Executive overview":
    m_cum, m_add = metric_cols()
    st.caption(f"Data source: {data_source}")

    # Explicit "as of" selector (clamped to latest available after future-date filter)
    min_asof = pd.to_datetime(cum["as_of_date"]).min().date() if len(cum) else latest_as_of
    as_of_sel = st.date_input("As of date", value=latest_as_of, min_value=min_asof, max_value=latest_as_of)

    # Work on a clamped view <= as_of_sel
    cum_eff = cum[cum["as_of_date"] <= pd.to_datetime(as_of_sel)].copy()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    # Biggest mover in the last 24h (based on daily adds for that as_of day)
    last_window = cum_eff[cum_eff["as_of_date"] == pd.to_datetime(as_of_sel)]
    if not last_window.empty:
        delta = last_window.sort_values(m_add, ascending=False).head(1)
        mover_date = delta["booking_date"].iloc[0].date()
        mover_val = int(delta[m_add].iloc[0]) if pd.notna(delta[m_add].iloc[0]) else 0
    else:
        mover_date, mover_val = None, 0

    # Next Fri/Sat vs weekday-median baseline at same lead (as of selected date)
    ref = pd.to_datetime(as_of_sel)
    fri = next_weekday(ref, 4)  # strictly next Friday
    sat = next_weekday(ref, 5)  # strictly next Saturday

    def get_cum(bd, asof):
        row = cum_eff[(cum_eff["booking_date"] == pd.to_datetime(bd)) & (cum_eff["as_of_date"] == pd.to_datetime(asof))]
        return float(row[m_cum].iloc[0]) if len(row) else np.nan

    fri_c = get_cum(fri, as_of_sel)
    sat_c = get_cum(sat, as_of_sel)

    # Baselines from entire history (median by DOW & lead)
    lead_fri = (pd.to_datetime(fri) - pd.to_datetime(as_of_sel)).days
    lead_sat = (pd.to_datetime(sat) - pd.to_datetime(as_of_sel)).days
    base = weekday_baseline(m_cum)
    fri_base = base[(base["booking_dow"] == "Friday") & (base["lookahead_days"] == lead_fri)][m_cum].squeeze() if not base.empty else np.nan
    sat_base = base[(base["booking_dow"] == "Saturday") & (base["lookahead_days"] == lead_sat)][m_cum].squeeze() if not base.empty else np.nan

    k1.metric("As of", as_of_sel.strftime("%Y-%m-%d"))
    k2.metric(f"Next Fri {fri.strftime('%Y-%m-%d')}", f"{int(fri_c):,}" if pd.notna(fri_c) else "—",
              delta=(f"{int(fri_c - fri_base):+,}" if (pd.notna(fri_c) and pd.notna(fri_base)) else None))
    k3.metric(f"Next Sat {sat.strftime('%Y-%m-%d')}", f"{int(sat_c):,}" if pd.notna(sat_c) else "—",
              delta=(f"{int(sat_c - sat_base):+,}" if (pd.notna(sat_c) and pd.notna(sat_base)) else None))
    k4.metric("Biggest mover last 24h", f"{mover_date} (+{mover_val:,})" if mover_date else "—")

    with st.expander("What am I looking at?"):
        st.markdown("""
- **Lookahead days** = days until the booking date (*booking_date − as_of_date*).
- **Cumulative** = total guests/bookings captured so far for that booking date (as of each day).
- **Daily adds** = new guests/bookings added on that day toward that booking date.
- **Baseline** = weekday **median** at the same lead (more robust than the mean).
""" )

    st.divider()

    # ---- Heatmap: % vs baseline, recent window, denser grid
    st.subheader("Where we’re over/under typical (percent vs baseline)")

    weeks = st.slider("Weeks to show", min_value=4, max_value=16, value=8, help="Window for booking_dates on the Y-axis.")
    cutoff_bd = pd.to_datetime(as_of_sel) - pd.Timedelta(days=7*weeks)

    ycol = m_cum if view == "Cumulative" else m_add

    tmp = cum_eff.copy()
    base = weekday_baseline(ycol)
    tmp = tmp.merge(base, on=["booking_dow", "lookahead_days"], suffixes=("", "_baseline"), how="left")
    denom = tmp[f"{ycol}_baseline"].replace({0: np.nan})
    tmp["pct_vs_baseline"] = (tmp[ycol] / denom) - 1.0

    # Only show recent booking_dates
    tmp = tmp[tmp["booking_date"] >= cutoff_bd]

    mat = tmp.groupby(["booking_date", "lookahead_days"])['pct_vs_baseline'].median().reset_index()

    if mat.empty:
        st.info("No data in the selected window.")
    else:
        max_lead = int(mat["lookahead_days"].max())
        all_leads = list(range(0, max_lead + 1))
        pivot = (mat.pivot(index="booking_date", columns="lookahead_days", values="pct_vs_baseline")
                    .reindex(columns=all_leads)
                    .interpolate(axis=1))

        vmax = float(np.nanpercentile(np.abs(pivot.values), 95)) if np.isfinite(pivot.values).any() else 0.5
        vmax = max(vmax, 0.1)

        fig = px.imshow(
            pivot, aspect="auto", origin="lower",
            color_continuous_scale="RdBu", zmin=-vmax, zmax=vmax
        )
        fig.update_layout(xaxis_title="Lookahead days", yaxis_title="Booking date")
        fig.update_yaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig, use_container_width=True)

        # Top under / over list (today's as_of selection)
        st.subheader("Top under / over vs baseline (today’s view)")
        today_slice = tmp[tmp["as_of_date"] == pd.to_datetime(as_of_sel)].copy()
        if not today_slice.empty:
            today_slice = today_slice.sort_values("pct_vs_baseline")
            under = today_slice.head(3).assign(**{"% vs base": (today_slice["pct_vs_baseline"]*100).round(0)})
            over  = today_slice.tail(3).assign(**{"% vs base": (today_slice["pct_vs_baseline"]*100).round(0)})
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most under**")
                st.dataframe(under[["booking_date","lookahead_days","% vs base"]], use_container_width=True)
            with col2:
                st.write("**Most over**")
                st.dataframe(over[["booking_date","lookahead_days","% vs base"]], use_container_width=True)

# ==============================
# Compare Dates
# ==============================
elif mode == "Compare dates":
    default = all_booking_dates[-2:] if len(all_booking_dates) >= 2 else all_booking_dates
    dates = st.multiselect("Pick up to four dates", all_booking_dates, default=default, format_func=fmt_date_opt, max_selections=4)
    align = st.radio("Align by", ["Lead time", "Calendar date"], index=0, horizontal=True)
    align_key = "lead" if align == "Lead time" else "calendar"
    data = curve_for_dates(dates, align=align_key)
    if data.empty:
        st.info("Select dates with curves available.")
    else:
        fig = px.line(data, x="x", y="y", color="series", markers=True)
        fig.update_layout(legend_title="Booking date", xaxis_title=("Lookahead days" if align_key=="lead" else "As of date"), yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# Compare DOW
# ==============================
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

# ==============================
# Upcoming Pacing
# ==============================
elif mode == "Upcoming pacing":
    st.subheader("Upcoming pacing")
    horizon = st.slider("Horizon days", 7, 42, 21)
    today = st.date_input("As of date", latest_as_of)
    pace = pacing_table(today=today, horizon=horizon)
    if pace.empty:
        st.info("No upcoming dates in range.")
    else:
        show = pace.copy()
        # Optional targets
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

        # Sort to surface risk first
        show = show.sort_values(["booking_date", "pace_index"])

        # Traffic-light styling
        def _color_pace(val):
            if pd.isna(val):
                return ""
            if val < 0.9:  return "background-color: #ffe5e5"  # red
            if val > 1.1:  return "background-color: #e6ffed"  # green
            return "background-color: #fff8e1"                 # amber

        styled = (
            show.style
                .applymap(_color_pace, subset=["pace_index"])
                .format({
                    "pace_index": "{:.2f}",
                    "current": "{:,.0f}",
                    "last_week_same_lead": "{:,.0f}",
                    "dow_median_same_lead": "{:.1f}",
                    "to_target": "{:,.0f}",
                })
        )
        st.dataframe(styled, use_container_width=True)

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

# ==============================
# Marketing vs Bookings (Lag Explorer)
# ==============================
else:
    st.subheader("Marketing vs bookings (daily, by transaction date)")
    mvb_df, mco_df, m_source = load_marketing()

    if mvb_df is None or mvb_df.empty:
        st.info(
            "No marketing files found. Expected marketing_vs_bookings.csv (and optional marketing_corr.csv) "
            f"in one of: {[str(p) for p in LOCAL_DIRS]} or via RAW_BASE."
        )
    else:
        mvb = mvb_df.copy()
        if "date" not in mvb.columns:
            st.error("marketing_vs_bookings.csv must include a 'date' column.")
            st.stop()
        mvb["date"] = pd.to_datetime(mvb["date"], errors="coerce")
        mvb = mvb.sort_values("date")

        # Identify IG columns
        ig_cols = [c for c in mvb.columns if c.startswith("ig_")]
        core_cols = [c for c in ["bookings", "guests"] if c in mvb.columns]

        if not core_cols:
            st.error("marketing_vs_bookings.csv must include 'bookings' or 'guests'.")
            st.stop()

        min_d, max_d = mvb["date"].min().date(), mvb["date"].max().date()
        start, end = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(start, (list, tuple)):
            start, end = start
        if start > end:
            start, end = end, start

        mvb_slice = mvb[(mvb["date"].dt.date >= start) & (mvb["date"].dt.date <= end)].copy()

        left_metric = st.radio("Bookings axis", core_cols, index=0, horizontal=True)
        default_igs = [c for c in ["ig_reach_7d","ig_clicks_7d","ig_impressions_7d"] if c in ig_cols][:2] or ig_cols[:2]
        picked_igs = st.multiselect("Pick IG series", ig_cols, default=default_igs)
        normalize = st.checkbox("Normalize series (z-score) on one axis", value=True)

        # Chart
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
                # Optional campaign overlays
                camps = _load_campaigns()
                if not camps.empty:
                    for _, row in camps.iterrows():
                        fig.add_vrect(x0=row["start_date"], x1=row["end_date"],
                                      fillcolor="#eef", opacity=0.25, line_width=0,
                                      annotation_text=str(row.get("label","campaign")),
                                      annotation_position="top left")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if len(picked_igs) == 1:
                    y2 = picked_igs[0]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=mvb_slice["date"], y=mvb_slice[left_metric], name=left_metric), secondary_y=False)
                    fig.add_trace(go.Scatter(x=mvb_slice["date"], y=mvb_slice[y2], name=y2), secondary_y=True)
                    fig.update_yaxes(title_text=left_metric, secondary_y=False)
                    fig.update_yaxes(title_text=y2, secondary_y=True)
                    # Overlays (if any)
                    camps = _load_campaigns()
                    if not camps.empty:
                        for _, row in camps.iterrows():
                            fig.add_vrect(x0=row["start_date"], x1=row["end_date"],
                                          fillcolor="#eef", opacity=0.25, line_width=0)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("For multiple IG series, enable normalization to compare on one axis.")

        # Lag Explorer
        st.subheader("Lag explorer")
        max_lag = st.slider("Test lags (days)", 0, 14, (0,7))
        l0, l1 = max_lag
        best_rows = []
        for ig in picked_igs or ig_cols:
            rows = []
            for L in range(l0, l1+1):
                shifted = mvb_slice[[left_metric, ig]].copy()
                shifted[ig] = shifted[ig].shift(L)
                sub = shifted.dropna()
                if len(sub) >= 3:
                    r = sub[left_metric].corr(sub[ig])
                else:
                    r = np.nan
                rows.append({"lag_days": L, "series": ig, "corr": r})
            part = pd.DataFrame(rows)
            if not part.empty and part["corr"].notna().any():
                best = part.loc[part["corr"].idxmax()]
                best_rows.append(best)
            if not part.empty:
                figL = px.bar(part, x="lag_days", y="corr", title=f"Correlation vs lag: {ig}")
                figL.update_layout(xaxis_title="Lag days (IG leading)", yaxis_title=f"corr({left_metric}, {ig} shifted)")
                st.plotly_chart(figL, use_container_width=True)

        # Best lag summary metric
        if best_rows:
            top = pd.DataFrame(best_rows).sort_values("corr", ascending=False).iloc[0]
            st.metric("Best correlation", f"{top['series']} @ {int(top['lag_days'])}d", delta=f"r={top['corr']:.2f}")

        # Correlations table
        st.subheader("Correlations (Pearson)")
        if mco_df is not None and not mco_df.empty:
            st.dataframe(mco_df, use_container_width=True)
        else:
            corr_cols = [left_metric] + (picked_igs or ig_cols[:2])
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

# ==============================
# Footer controls
# ==============================
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.experimental_rerun()
with colB:
    st.caption(f"Data source: {data_source}")
with colC:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

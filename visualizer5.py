# visualizer5.py
"""
Bookings & Marketing Overview (Streamlit)

Key features:
- Executive Overview pacing vs typical (Fri/Sat KPIs, heatmap vs baseline)
- Compare specific booking dates or weekday patterns
- Upcoming pacing table vs last week and vs typical
- Marketing vs bookings + lag explorer
- Booking Method filter (online vs phone etc.), if by-method snapshot is available
- "As of date" selector now lets you choose today even if latest ingest is yesterday.
"""

from pathlib import Path
from datetime import datetime, date
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

REQUIRED_FILES = [
    "cumulative_asof_snapshot.csv",
    "daily_snapshot.csv",
]

# ==============================
# Local timezone helper
# ==============================
try:
    from zoneinfo import ZoneInfo  # py3.9+
    CT = ZoneInfo("America/Chicago")
except Exception:
    CT = None

def today_local():
    """Return today's date in America/Chicago (if available)."""
    return (datetime.now(CT).date() if CT else datetime.now().date())

# ==============================
# Query params (for deep links)
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
    """See if all required CSVs are in a known local folder."""
    for d in LOCAL_DIRS:
        if all((d / f).exists() for f in REQUIRED_FILES):
            return d
    return None

def read_from_raw_base():
    """If RAW_BASE is set (Streamlit Cloud), pull CSVs from GitHub raw URLs."""
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
    """Fallback: manual uploads if nothing else is available."""
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
    """
    Load:
      - cumulative_asof_snapshot.csv
      - daily_snapshot.csv
      - week_over_week_latest.csv (optional)
    Prefer local, then RAW_BASE (GitHub raw), then upload.
    """
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

def _load_campaigns():
    """Optional overlay (campaign flight windows)."""
    for d in LOCAL_DIRS:
        p = d / "campaigns.csv"
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=["start_date","end_date"])
            except Exception:
                pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_by_method_snapshot():
    """
    Load cumulative_asof_by_method.csv which has:
    booking_date, booking_method, as_of_date, cum_guests, cum_bookings
    """
    for d in LOCAL_DIRS:
        p = d / "cumulative_asof_by_method.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["booking_date", "as_of_date"])
    if RAW_BASE:
        try:
            return pd.read_csv(
                f"{RAW_BASE}/cumulative_asof_by_method.csv",
                parse_dates=["booking_date", "as_of_date"]
            )
        except Exception:
            pass
    return None

# ==============================
# Utilities
# ==============================
def fmt_date_opt(x):
    """Format date-like values safely."""
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
    """Take a pandas Series of datetimes, return sorted list of Python date objects."""
    return (
        pd.to_datetime(series, errors="coerce")
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

def zscore(col: pd.Series) -> pd.Series:
    """Normalize column to mean 0/std 1 for overlay plotting."""
    col = pd.to_numeric(col, errors="coerce")
    mu = col.mean()
    sd = col.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return col * 0
    return (col - mu) / sd

def next_weekday(dts, weekday_int):
    """
    Return the next occurrence of weekday_int after dts (0=Mon..6=Sun).
    'Next' never means 'today'.
    """
    days_ahead = (weekday_int - dts.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (dts + pd.Timedelta(days=days_ahead)).date()

# ==============================
# Transform core data
# ==============================
cum_raw, daily_raw, wow_raw, data_source = load_core_data()

# Parse key dates
for col in ["booking_date", "as_of_date"]:
    if col in cum_raw.columns:
        cum_raw[col] = pd.to_datetime(cum_raw[col], errors="coerce")
for col in ["transaction_date", "booking_date"]:
    if col in daily_raw.columns:
        daily_raw[col] = pd.to_datetime(daily_raw[col], errors="coerce")

# Ensure required numeric cols exist
for col in ["cum_guests", "cum_bookings"]:
    if col not in cum_raw.columns:
        cum_raw[col] = np.nan
for col in ["guests", "bookings"]:
    if col not in daily_raw.columns:
        daily_raw[col] = np.nan

# Build cumulative frame and daily adds
cum = cum_raw.sort_values(["booking_date", "as_of_date"]).copy()

cum["daily_add_guests"] = (
    cum.groupby("booking_date")["cum_guests"]
       .diff()
       .fillna(cum["cum_guests"])
)
cum["daily_add_bookings"] = (
    cum.groupby("booking_date")["cum_bookings"]
       .diff()
       .fillna(cum["cum_bookings"])
)

# Derive lookahead and weekday
cum["as_of_date"] = pd.to_datetime(cum["as_of_date"], errors="coerce")
cum["booking_date"] = pd.to_datetime(cum["booking_date"], errors="coerce")
cum["lookahead_days"] = (cum["booking_date"] - cum["as_of_date"]).dt.days
cum["booking_dow"] = cum["booking_date"].dt.day_name()

# Clamp to "today" so we never treat future dates as already ingested
TODAY = today_local()
cum = cum[cum["as_of_date"].dt.date <= TODAY].copy()

# last snapshot we actually HAVE in data
latest_as_of_ts = pd.to_datetime(cum["as_of_date"]).max()
latest_as_of = (latest_as_of_ts.date() if pd.notna(latest_as_of_ts) else TODAY)

# UI can go up to today's date, even if latest_as_of is yesterday
DISPLAY_MAX_ASOF = max(latest_as_of, TODAY)

# ==============================
# Sidebar / controls
# ==============================
cum_by_method = load_by_method_snapshot()
method_options = (
    sorted(cum_by_method["booking_method"].dropna().unique().tolist())
    if (cum_by_method is not None
        and not cum_by_method.empty
        and "booking_method" in cum_by_method.columns)
    else []
)

_mode_opts = [
    "Executive overview",
    "Compare dates",
    "Compare day of week",
    "Upcoming pacing",
    "Marketing vs bookings",
]
_mode_qs = (qs.get("mode") or ["Executive_overview"])[0].replace("_", " ")
_mode_idx = _mode_opts.index(_mode_qs) if _mode_qs in _mode_opts else 0

_metric_qs = (qs.get("metric") or ["Guests"])[0]
_metric_idx = 0 if _metric_qs == "Guests" else 1

_view_qs = (qs.get("view") or ["Cumulative"])[0].replace("_", " ")
_view_idx = 0 if _view_qs == "Cumulative" else 1

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", _mode_opts, index=_mode_idx)
metric = st.sidebar.radio("Metric", ["Guests", "Bookings"], index=_metric_idx)
view = st.sidebar.radio("View", ["Cumulative", "Daily adds"], index=_view_idx)

picked_methods = []
if method_options:
    picked_methods = st.sidebar.multiselect(
        "Booking method",
        method_options,
        default=method_options
    )

# Optional live debugging to confirm we loaded method data
with st.sidebar.expander("Data status", expanded=False):
    st.caption(f"Core source: {data_source}")
    if cum_by_method is None or cum_by_method.empty or "booking_method" not in cum_by_method.columns:
        st.error("Booking method snapshot: NOT FOUND")
    else:
        methods_list = sorted(cum_by_method["booking_method"].dropna().unique().tolist())
        st.success(f"Booking method snapshot: FOUND ({len(methods_list)} methods)")
        st.write(", ".join(methods_list[:12]) + ("…" if len(methods_list) > 12 else ""))

# Keep URL sharable
_set_params(
    mode=mode.replace(" ", "_"),
    metric=metric,
    view=view.replace(" ", "_"),
)

def metric_cols(prefix=""):
    if metric == "Guests":
        return f"{prefix}cum_guests", f"{prefix}daily_add_guests"
    return f"{prefix}cum_bookings", f"{prefix}daily_add_bookings"

# If the user picked methods, swap cum with the aggregated-by-method view
if cum_by_method is not None and method_options and picked_methods:
    cm = cum_by_method[cum_by_method["booking_method"].isin(picked_methods)].copy()

    # collapse across chosen methods
    cm = (
        cm.groupby(["booking_date", "as_of_date"], as_index=False)[["cum_guests", "cum_bookings"]]
          .sum()
    )

    cm = cm.sort_values(["booking_date", "as_of_date"])
    cm["daily_add_guests"] = (
        cm.groupby("booking_date")["cum_guests"]
          .diff()
          .fillna(cm["cum_guests"])
    )
    cm["daily_add_bookings"] = (
        cm.groupby("booking_date")["cum_bookings"]
          .diff()
          .fillna(cm["cum_bookings"])
    )
    cm["booking_dow"] = pd.to_datetime(cm["booking_date"]).dt.day_name()
    cm["lookahead_days"] = (
        pd.to_datetime(cm["booking_date"]) - pd.to_datetime(cm["as_of_date"])
    ).dt.days

    cm["as_of_date"] = pd.to_datetime(cm["as_of_date"], errors="coerce")
    cm = cm[cm["as_of_date"].dt.date <= TODAY].copy()

    # Now recalc latest_as_of / DISPLAY_MAX_ASOF inside method filter path
    latest_asof_ts_f = pd.to_datetime(cm["as_of_date"]).max()
    latest_as_of_f = (latest_asof_ts_f.date() if pd.notna(latest_asof_ts_f) else TODAY)
    DISPLAY_MAX_ASOF = max(latest_as_of_f, TODAY)
    latest_as_of = latest_as_of_f

    cum = cm

# After method filtering, recompute the list of booking dates for pickers
all_booking_dates = to_py_dates(cum["booking_date"])

def weekday_baseline(metric_key: str):
    """Median baseline by DOW and leadtime."""
    base = (
        cum.groupby(["booking_dow", "lookahead_days"])[metric_key]
           .median()
           .reset_index()
    )
    return base

# ==============================
# Helpers for plotting / tables
# ==============================
def curve_for_dates(dates, align="lead"):
    """
    Build a long-form DF for plotting one or more booking_date curves
    either aligned by lookahead_days ("lead") or by as_of_date ("calendar").
    """
    m_cum, m_add = metric_cols()
    out = []
    for d in dates:
        d_ts = pd.to_datetime(d)
        sub = cum[cum["booking_date"] == d_ts].copy()
        if sub.empty:
            continue
        ycol = m_cum if view == "Cumulative" else m_add
        xcol = "lookahead_days" if align == "lead" else "as_of_date"
        if ycol not in sub.columns or xcol not in sub.columns:
            continue
        sub = sub.sort_values(xcol)
        sub["series"] = fmt_date_opt(d)
        out.append(
            sub[[xcol, ycol, "series"]]
            .rename(columns={xcol: "x", ycol: "y"})
        )
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["x", "y", "series"])

def dow_profile(day_name, q_low=0.1, q_high=0.9):
    """
    Median + 10/90% band for a given weekday (e.g. 'Friday') across all dates.
    """
    m_cum, m_add = metric_cols()
    ycol = m_cum if view == "Cumulative" else m_add
    tmp = cum[cum["booking_dow"] == day_name]

    prof = (
        tmp.groupby("lookahead_days")[ycol]
           .median()
           .reset_index(name="median")
    )
    band = (
        tmp.groupby("lookahead_days")[ycol]
           .quantile([q_low, q_high])
           .unstack()
           .reset_index()
    )
    band.columns = ["lookahead_days", "p_low", "p_high"]
    prof = prof.rename(columns={"lookahead_days": "x"})
    band = band.rename(columns={"lookahead_days": "x"})
    return prof, band

def pacing_table(today=None, horizon=21):
    """
    For horizon N days forward, show:
      - current cumulative
      - last week's cumulative at same lead
      - median baseline at same weekday+lead
      - pace index vs median baseline
    """
    if today is None:
        today = latest_as_of
    m_cum, _ = metric_cols()

    future_dates = [
        d for d in all_booking_dates
        if d >= today and (d - today).days <= horizon
    ]

    rows = []
    for bd in future_dates:
        cur = cum[
            (cum["booking_date"] == pd.to_datetime(bd)) &
            (cum["as_of_date"] == pd.to_datetime(today))
        ]
        cur_val = cur[m_cum].iloc[0] if len(cur) else np.nan

        lead = (pd.to_datetime(bd) - pd.to_datetime(today)).days

        last_week_asof = pd.to_datetime(today) - pd.Timedelta(days=7)
        last_week_bd = pd.to_datetime(bd) - pd.Timedelta(days=7)

        prev = cum[
            (cum["booking_date"] == last_week_bd) &
            (cum["as_of_date"] == last_week_asof)
        ]
        prev_val = prev[m_cum].iloc[0] if len(prev) else np.nan

        dow = pd.to_datetime(bd).day_name()
        avg = (
            cum[cum["booking_dow"] == dow]
            .groupby("lookahead_days")[m_cum]
            .median()
        )
        avg_val = avg.get(lead, np.nan)

        pace = (
            cur_val / avg_val
            if pd.notna(cur_val) and pd.notna(avg_val) and avg_val != 0
            else np.nan
        )

        rows.append({
            "booking_date": bd,
            "lookahead_days": int(lead),
            "current": float(cur_val) if pd.notna(cur_val) else np.nan,
            "last_week_same_lead": float(prev_val) if pd.notna(prev_val) else np.nan,
            "dow_median_same_lead": float(avg_val) if pd.notna(avg_val) else np.nan,
            "pace_index": float(pace) if pd.notna(pace) else np.nan,
        })

    return pd.DataFrame(rows)

# ==============================
# MODE: Executive overview
# ==============================
if mode == "Executive overview":
    m_cum, m_add = metric_cols()
    st.caption(f"Data source: {data_source}")

    # Let the user *select* up to DISPLAY_MAX_ASOF (today),
    # even if latest_as_of is earlier. But calculations will
    # still use only data available as of the last ingest.
    min_asof = (
        pd.to_datetime(cum["as_of_date"]).min().date()
        if len(cum) else latest_as_of
    )

    as_of_sel = st.date_input(
        "As of date",
        value=DISPLAY_MAX_ASOF,
        min_value=min_asof,
        max_value=DISPLAY_MAX_ASOF,
        help="Shown date can be 'today' even if last ingest was yesterday. Metrics use last ingested day."
    )

    # analysis_cutoff = min(user-picked date, actual last ingest)
    analysis_cutoff = min(as_of_sel, latest_as_of)

    # Filter data to everything on/before analysis_cutoff
    cum_eff = cum[cum["as_of_date"] <= pd.to_datetime(analysis_cutoff)].copy()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)

    # Biggest mover last 24h = which booking_date got the most adds on the cutoff day
    last_window = cum_eff[cum_eff["as_of_date"] == pd.to_datetime(analysis_cutoff)]
    if not last_window.empty:
        delta = last_window.sort_values(m_add, ascending=False).head(1)
        mover_date = delta["booking_date"].iloc[0].date()
        mover_val = int(delta[m_add].iloc[0]) if pd.notna(delta[m_add].iloc[0]) else 0
    else:
        mover_date, mover_val = None, 0

    # Next Fri / Sat KPIs
    ref = pd.to_datetime(analysis_cutoff)
    next_fri = next_weekday(ref, 4)  # strictly next Friday
    next_sat = (pd.to_datetime(next_fri) + pd.Timedelta(days=1)).date()  # matching Saturday

    def get_cum(bd, asof):
        row = cum_eff[
            (cum_eff["booking_date"] == pd.to_datetime(bd)) &
            (cum_eff["as_of_date"] == pd.to_datetime(asof))
        ]
        return float(row[m_cum].iloc[0]) if len(row) else np.nan

    fri_c = get_cum(next_fri, analysis_cutoff)
    sat_c = get_cum(next_sat, analysis_cutoff)

    # Baseline medians for same weekday & same lead
    lead_fri = (pd.to_datetime(next_fri) - pd.to_datetime(analysis_cutoff)).days
    lead_sat = (pd.to_datetime(next_sat) - pd.to_datetime(analysis_cutoff)).days
    base_tbl = weekday_baseline(m_cum)

    fri_base = base_tbl[
        (base_tbl["booking_dow"] == "Friday") &
        (base_tbl["lookahead_days"] == lead_fri)
    ][m_cum].squeeze() if not base_tbl.empty else np.nan

    sat_base = base_tbl[
        (base_tbl["booking_dow"] == "Saturday") &
        (base_tbl["lookahead_days"] == lead_sat)
    ][m_cum].squeeze() if not base_tbl.empty else np.nan

    k1.metric(
        "As of (analysis)",
        analysis_cutoff.strftime("%Y-%m-%d")
    )
    k2.metric(
        f"Next Fri {next_fri.strftime('%Y-%m-%d')}",
        f"{int(fri_c):,}" if pd.notna(fri_c) else "—",
        delta=(f"{int(fri_c - fri_base):+,}"
               if (pd.notna(fri_c) and pd.notna(fri_base))
               else None)
    )
    k3.metric(
        f"Next Sat {next_sat.strftime('%Y-%m-%d')}",
        f"{int(sat_c):,}" if pd.notna(sat_c) else "—",
        delta=(f"{int(sat_c - sat_base):+,}"
               if (pd.notna(sat_c) and pd.notna(sat_base))
               else None)
    )
    k4.metric(
        "Biggest mover last 24h",
        f"{mover_date} (+{mover_val:,})" if mover_date else "—"
    )

    with st.expander("What am I looking at?"):
        st.markdown(
            """
- **Lookahead days** = days until the booking date (*booking_date − as_of_date*).
- **Cumulative** = total guests / bookings captured so far for that booking date.
- **Daily adds** = how many new guests / bookings were added on that specific as_of_date toward that booking date.
- **Baseline** = weekday median at the same lead (robust typical performance for that weekday).
- **As of date selector** = You can pick 'today'. If today's ingest isn't in yet, math uses the latest ingest (yesterday).
            """
        )

    st.divider()

    # Heatmap: % vs baseline for recent booking_dates
    st.subheader("Where we’re over / under typical (% vs baseline)")
    weeks = st.slider(
        "Weeks to show",
        min_value=4,
        max_value=16,
        value=8,
        help="Window for booking_dates on the Y-axis."
    )

    cutoff_bd = pd.to_datetime(analysis_cutoff) - pd.Timedelta(days=7 * weeks)
    ycol = m_cum if view == "Cumulative" else m_add

    tmp = cum_eff.copy()
    base_for_y = weekday_baseline(ycol)
    tmp = tmp.merge(
        base_for_y,
        on=["booking_dow", "lookahead_days"],
        suffixes=("", "_baseline"),
        how="left"
    )

    denom = tmp[f"{ycol}_baseline"].replace({0: np.nan})
    tmp["pct_vs_baseline"] = (tmp[ycol] / denom) - 1.0

    # only recent booking_dates
    tmp = tmp[tmp["booking_date"] >= cutoff_bd]

    mat = (
        tmp.groupby(["booking_date", "lookahead_days"])["pct_vs_baseline"]
           .median()
           .reset_index()
    )

    if mat.empty:
        st.info("No data in the selected window.")
    else:
        max_lead = int(mat["lookahead_days"].max())
        all_leads = list(range(0, max_lead + 1))

        pivot = (
            mat.pivot(index="booking_date", columns="lookahead_days", values="pct_vs_baseline")
               .reindex(columns=all_leads)
               .interpolate(axis=1)
        )

        # scale symmetric around 0 using 95th pct
        if np.isfinite(pivot.values).any():
            vmax = float(np.nanpercentile(np.abs(pivot.values), 95))
        else:
            vmax = 0.5
        vmax = max(vmax, 0.1)

        fig = px.imshow(
            pivot,
            aspect="auto",
            origin="lower",
            color_continuous_scale="RdBu",
            zmin=-vmax,
            zmax=vmax
        )
        fig.update_layout(
            xaxis_title="Lookahead days",
            yaxis_title="Booking date"
        )
        fig.update_yaxes(tickformat="%Y-%m-%d")

        st.plotly_chart(fig, use_container_width=True)

        # Callout table: top under / over today
        st.subheader("Top under / over vs baseline (latest day shown)")
        today_slice = tmp[tmp["as_of_date"] == pd.to_datetime(analysis_cutoff)].copy()
        if not today_slice.empty:
            today_slice = today_slice.sort_values("pct_vs_baseline")
            under = today_slice.head(3).assign(
                **{"% vs base": (today_slice["pct_vs_baseline"] * 100).round(0)}
            )
            over = today_slice.tail(3).assign(
                **{"% vs base": (today_slice["pct_vs_baseline"] * 100).round(0)}
            )
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most under**")
                st.dataframe(
                    under[["booking_date", "lookahead_days", "% vs base"]],
                    use_container_width=True
                )
            with col2:
                st.write("**Most over**")
                st.dataframe(
                    over[["booking_date", "lookahead_days", "% vs base"]],
                    use_container_width=True
                )

# ==============================
# MODE: Compare dates
# ==============================
elif mode == "Compare dates":
    default = all_booking_dates[-2:] if len(all_booking_dates) >= 2 else all_booking_dates
    dates = st.multiselect(
        "Pick up to four booking dates",
        all_booking_dates,
        default=default,
        format_func=fmt_date_opt,
        max_selections=4,
    )
    align = st.radio(
        "Align by",
        ["Lead time", "Calendar date"],
        index=0,
        horizontal=True
    )
    align_key = "lead" if align == "Lead time" else "calendar"
    data = curve_for_dates(dates, align=align_key)

    if data.empty:
        st.info("Select dates with curves available.")
    else:
        fig = px.line(data, x="x", y="y", color="series", markers=True)
        fig.update_layout(
            legend_title="Booking date",
            xaxis_title=("Lookahead days" if align_key == "lead" else "As of date"),
            yaxis_title=metric,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# MODE: Compare day of week
# ==============================
elif mode == "Compare day of week":
    dows = sorted([d for d in cum["booking_dow"].dropna().unique().tolist()])
    dow = st.selectbox("Day of week", dows, index=0)

    choices = to_py_dates(
        cum.loc[cum["booking_date"].dt.day_name() == dow, "booking_date"]
    )
    ref = st.selectbox(
        "Reference booking date",
        choices,
        index=len(choices) - 1 if choices else 0,
        format_func=fmt_date_opt,
    )

    prof, band = dow_profile(dow)
    ref_curve = curve_for_dates([ref], align="lead")

    st.subheader(f"{dow} profile vs {fmt_date_opt(ref)}")

    fig = px.line(ref_curve, x="x", y="y", color="series", markers=True)
    if not prof.empty:
        fig.add_traces(
            px.line(prof, x="x", y="median")
              .update_traces(name=f"Median {dow}")
              .data
        )
    if not band.empty:
        fig.add_traces(
            px.area(band, x="x", y="p_high")
              .update_traces(name="90th pct")
              .data
        )
        fig.add_traces(
            px.area(band, x="x", y="p_low")
              .update_traces(name="10th pct")
              .data
        )
    fig.update_layout(
        xaxis_title="Lookahead days",
        yaxis_title=metric,
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# MODE: Upcoming pacing
# ==============================
elif mode == "Upcoming pacing":
    st.subheader("Upcoming pacing vs typical")

    horizon = st.slider("Horizon days", 7, 42, 21)
    today_pick = st.date_input(
        "As of date for pacing table",
        value=latest_as_of,
        help="Use a recent ingest date to compare lead vs typical and vs last week."
    )

    pace = pacing_table(today=today_pick, horizon=horizon)

    if pace.empty:
        st.info("No upcoming dates in range.")
    else:
        show = pace.copy()

        # Optional target file (not required)
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

        # Sort to float the worst pacing up
        show = show.sort_values(["booking_date", "pace_index"])

        # Traffic light styling for pace_index
        def _color_pace(val):
            if pd.isna(val):
                return ""
            if val < 0.9:
                return "background-color: #ffe5e5"  # red
            if val > 1.1:
                return "background-color: #e6ffed"  # green
            return "background-color: #fff8e1"      # amber

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
            fig3.update_layout(
                xaxis_title="Booking date",
                yaxis_title=f"Delta {metric}",
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            "Download pacing table CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"pacing_{today_pick}.csv",
            mime="text/csv",
        )

# ==============================
# MODE: Marketing vs bookings
# ==============================
else:
    st.subheader("Marketing vs bookings (daily, by transaction date)")

    mvb_df, mco_df, m_source = load_marketing()

    if mvb_df is None or mvb_df.empty:
        st.info(
            "No marketing files found. Expected marketing_vs_bookings.csv "
            "(and optional marketing_corr.csv) in booking_outputs/ or via RAW_BASE."
        )
    else:
        mvb = mvb_df.copy()

        if "date" not in mvb.columns:
            st.error("marketing_vs_bookings.csv must include a 'date' column.")
            st.stop()

        mvb["date"] = pd.to_datetime(mvb["date"], errors="coerce")
        mvb = mvb.sort_values("date")

        ig_cols = [c for c in mvb.columns if c.startswith("ig_")]
        core_cols = [c for c in ["bookings", "guests"] if c in mvb.columns]

        if not core_cols:
            st.error("marketing_vs_bookings.csv must include 'bookings' or 'guests'.")
            st.stop()

        min_d = mvb["date"].min().date()
        max_d = mvb["date"].max().date()

        start, end = st.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            help="Filter the marketing and booking timeseries window.",
        )
        # Streamlit returns a tuple for date_input(range), but we guard:
        if isinstance(start, (list, tuple)):
            start, end = start
        if start > end:
            start, end = end, start

        mvb_slice = mvb[
            (mvb["date"].dt.date >= start) &
            (mvb["date"].dt.date <= end)
        ].copy()

        left_metric = st.radio(
            "Bookings axis (left)",
            core_cols,
            index=0,
            horizontal=True,
        )

        default_igs = [
            c for c in ["ig_reach_7d", "ig_clicks_7d", "ig_impressions_7d"]
            if c in ig_cols
        ][:2] or ig_cols[:2]

        picked_igs = st.multiselect(
            "Pick IG series",
            ig_cols,
            default=default_igs,
        )

        normalize = st.checkbox(
            "Normalize all series (z-score) on one axis",
            value=True,
        )

        # CHART
        if len(picked_igs) == 0:
            st.info("Select at least one IG series.")
        else:
            if normalize:
                plot_df = mvb_slice[["date", left_metric] + picked_igs].copy()
                for c in [left_metric] + picked_igs:
                    plot_df[c] = zscore(plot_df[c])

                long_df = plot_df.melt(
                    id_vars="date",
                    value_vars=[left_metric] + picked_igs,
                    var_name="series",
                    value_name="z",
                )
                fig = px.line(long_df, x="date", y="z", color="series")
                fig.update_layout(yaxis_title="Z-score (normalized)")

                camps = _load_campaigns()
                if not camps.empty:
                    for _, row in camps.iterrows():
                        fig.add_vrect(
                            x0=row["start_date"],
                            x1=row["end_date"],
                            fillcolor="#eef",
                            opacity=0.25,
                            line_width=0,
                            annotation_text=str(row.get("label","campaign")),
                            annotation_position="top left",
                        )

                st.plotly_chart(fig, use_container_width=True)

            else:
                # Dual-axis if they pick exactly one IG metric
                if len(picked_igs) == 1:
                    y2 = picked_igs[0]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(
                            x=mvb_slice["date"],
                            y=mvb_slice[left_metric],
                            name=left_metric,
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=mvb_slice["date"],
                            y=mvb_slice[y2],
                            name=y2,
                        ),
                        secondary_y=True,
                    )
                    fig.update_yaxes(title_text=left_metric, secondary_y=False)
                    fig.update_yaxes(title_text=y2, secondary_y=True)

                    camps = _load_campaigns()
                    if not camps.empty:
                        for _, row in camps.iterrows():
                            fig.add_vrect(
                                x0=row["start_date"],
                                x1=row["end_date"],
                                fillcolor="#eef",
                                opacity=0.25,
                                line_width=0,
                            )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("For multiple IG series, enable normalization to compare on one axis.")

        # Lag Explorer
        st.subheader("Lag explorer (IG leading bookings)")
        max_lag = st.slider(
            "Test lag range (days IG leads bookings)",
            0,
            14,
            (0, 7),
        )
        l0, l1 = max_lag
        best_rows = []

        for ig in (picked_igs or ig_cols):
            rows = []
            for L in range(l0, l1 + 1):
                shifted = mvb_slice[[left_metric, ig]].copy()
                shifted[ig] = shifted[ig].shift(L)
                sub = shifted.dropna()
                if len(sub) >= 3:
                    r = sub[left_metric].corr(sub[ig])
                else:
                    r = np.nan
                rows.append({
                    "lag_days": L,
                    "series": ig,
                    "corr": r,
                })
            part = pd.DataFrame(rows)

            if not part.empty and part["corr"].notna().any():
                best = part.loc[part["corr"].idxmax()]
                best_rows.append(best)

            if not part.empty:
                figL = px.bar(
                    part,
                    x="lag_days",
                    y="corr",
                    title=f"Correlation vs lag: {ig}",
                )
                figL.update_layout(
                    xaxis_title="Lag days (IG shifted forward)",
                    yaxis_title=f"corr({left_metric}, {ig} shifted)",
                )
                st.plotly_chart(figL, use_container_width=True)

        if best_rows:
            top = (
                pd.DataFrame(best_rows)
                .sort_values("corr", ascending=False)
                .iloc[0]
            )
            st.metric(
                "Best correlation",
                f"{top['series']} @ {int(top['lag_days'])}d",
                delta=f"r={top['corr']:.2f}",
            )

        st.subheader("Correlations (Pearson)")
        mvb_numeric_cols = mvb_slice.select_dtypes(include=[np.number]).columns.tolist()

        # If you uploaded marketing_corr.csv, show that instead
        mvb_corr_df = None
        mvb_loaded_corr = try_load_marketing_from_local() or try_load_marketing_from_raw()
        # mvb_loaded_corr is (mvb_df, mco_df, source) OR None
        # but we already consumed mvb_df, so check mco_df
        # we don't re-call in the block to avoid confusion, just reuse mco_df we loaded at top:
        # mco_df is from load_marketing()

        if mco_df is not None and not mco_df.empty:
            mvb_corr_df = mco_df.copy()

        if mvb_corr_df is not None:
            st.dataframe(mvb_corr_df, use_container_width=True)
        else:
            # compute quick corr on-the-fly
            # only show cols that actually exist in this slice
            corr_cols = [c for c in mvb_numeric_cols if c in ([left_metric] + picked_igs)]
            sub = mvb_slice[corr_cols].dropna()
            if sub.shape[0] >= 3 and len(corr_cols) >= 2:
                corr = sub.corr().round(3)
                st.dataframe(corr, use_container_width=True)
            else:
                st.info("Not enough rows in the selected date range to compute correlation.")

        st.download_button(
            "Download filtered marketing_vs_bookings.csv",
            data=mvb_slice.to_csv(index=False).encode("utf-8"),
            file_name=f"marketing_vs_bookings_{start}_to_{end}.csv",
            mime="text/csv",
        )

# ==============================
# Footer controls
# ==============================
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.experimental_rerun()
with colB:
    st.caption(f"Data source: {data_source}")
with colC:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

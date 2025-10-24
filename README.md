# Brass Monkey Booking Visualizer

Simple Streamlit app for visualizing booking lead time, day-of-week patterns, and pacing.

## App scripts
- `visualizer5.py` — Streamlit dashboard
- `booking_lead_analysis.py` — data prep to produce `booking_outputs/*.csv`
- `make_report.py` — optional static HTML report

## Data folder
Place these files under `booking_outputs/`:
- `cumulative_asof_snapshot.csv`
- `daily_snapshot.csv`
- `week_over_week_latest.csv` (optional)

## Local run
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run visualizer5.py --server.runOnSave true
```

## Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, connect GitHub, select your repo, script path `visualizer5.py`.
3. Add `requirements.txt` as provided.
4. Deploy.

## Notes
- Keep private CSVs out of the repo. Commit anonymized samples if you need a demo.

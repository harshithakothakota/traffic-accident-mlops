"""ETL pipeline for UK traffic accidents (2023) + historical weather.

Each function below is ONE step of the pipeline (one Airflow task).
Steps pass data to each other through files in the data/ folder:

    data/raw/       downloaded files, never modified
    data/staging/   cleaned + standardised tables
    data/rejected/  rows that failed validation (with the reason)

Run everything without Airflow:   python etl.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

YEAR = 2023
RAW = ROOT / "data" / "raw"
STAGING = ROOT / "data" / "staging"
REJECTED = ROOT / "data" / "rejected"
for folder in (RAW, STAGING, REJECTED):
    folder.mkdir(parents=True, exist_ok=True)

STATS19_URL = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{table}-{year}.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# STATS19 police force codes -> names (from the DfT data guide)
POLICE_FORCES = {
    1: "Metropolitan Police", 3: "Cumbria", 4: "Lancashire", 5: "Merseyside",
    6: "Greater Manchester", 7: "Cheshire", 10: "Northumbria", 11: "Durham",
    12: "North Yorkshire", 13: "West Yorkshire", 14: "South Yorkshire",
    16: "Humberside", 17: "Cleveland", 20: "West Midlands", 21: "Staffordshire",
    22: "West Mercia", 23: "Warwickshire", 30: "Derbyshire", 31: "Nottinghamshire",
    32: "Lincolnshire", 33: "Leicestershire", 34: "Northamptonshire",
    35: "Cambridgeshire", 36: "Norfolk", 37: "Suffolk", 40: "Bedfordshire",
    41: "Hertfordshire", 42: "Essex", 43: "Thames Valley", 44: "Hampshire",
    45: "Surrey", 46: "Kent", 47: "Sussex", 48: "City of London",
    50: "Devon and Cornwall", 52: "Avon and Somerset", 53: "Gloucestershire",
    54: "Wiltshire", 55: "Dorset", 60: "North Wales", 61: "Gwent",
    62: "South Wales", 63: "Dyfed-Powys", 99: "Police Scotland",
}


def engine():
    """Database connection. The URL (with password) comes from the .env file."""
    return create_engine(os.environ["DATABASE_URL"])


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------- 1. EXTRACT
def extract_accident_data():
    """Download the 3 STATS19 files for 2023 (collisions, vehicles, casualties)."""
    for table in ("collision", "vehicle", "casualty"):
        path = RAW / f"{table}_{YEAR}.csv"
        if path.exists():
            log(f"{path.name} already downloaded - skipping")
            continue
        url = STATS19_URL.format(table=table, year=YEAR)
        for attempt in range(3):                      # retry up to 3 times
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
                break
            except requests.RequestException as error:
                log(f"download failed ({error}), retrying...")
                time.sleep(5)
        else:
            raise RuntimeError(f"could not download {url}")
        rows = len(pd.read_csv(path, usecols=[0]))
        log(f"downloaded {path.name}: {rows:,} rows")


def extract_weather_data():
    """Download hourly 2023 weather from Open-Meteo.

    The UK is split into 1 x 1 degree grid squares. For every square with at
    least 50 accidents we download the weather at the centre of the square."""
    path = RAW / f"weather_{YEAR}.csv"
    if path.exists():
        log(f"{path.name} already downloaded - skipping")
        return
    accidents = pd.read_csv(RAW / f"collision_{YEAR}.csv", usecols=["latitude", "longitude"])
    accidents = accidents.apply(pd.to_numeric, errors="coerce").dropna()
    accidents["cell_lat"] = np.floor(accidents.latitude) + 0.5
    accidents["cell_lon"] = np.floor(accidents.longitude) + 0.5
    cells = accidents.groupby(["cell_lat", "cell_lon"]).size().reset_index(name="n")
    cells = cells[cells.n >= 50]
    log(f"downloading weather for {len(cells)} grid squares")

    frames = []
    for cell in cells.itertuples():
        params = {
            "latitude": cell.cell_lat, "longitude": cell.cell_lon,
            "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "Europe/London",
        }
        for attempt in range(5):
            try:
                response = requests.get(WEATHER_URL, params=params, timeout=90)
                response.raise_for_status()
                break
            except requests.RequestException as error:
                log(f"weather request failed ({error}), retrying...")
                time.sleep(30)
        else:
            log(f"giving up on cell {cell.cell_lat},{cell.cell_lon}")
            continue
        hourly = pd.DataFrame(response.json()["hourly"])
        hourly["cell_lat"], hourly["cell_lon"] = cell.cell_lat, cell.cell_lon
        frames.append(hourly)
        time.sleep(2)                                   # be polite to the free API
    weather = pd.concat(frames)
    weather.to_csv(path, index=False)
    log(f"saved {len(weather):,} hourly weather rows")


# ----------------------------------------------------------------- 2. STAGE
def stage_collisions():
    """Keep the useful collision columns and fix their data types."""
    df = pd.read_csv(RAW / f"collision_{YEAR}.csv", low_memory=False)
    df = df[["collision_index", "date", "time", "latitude", "longitude",
             "police_force", "local_authority_ons_district", "collision_severity",
             "number_of_vehicles", "number_of_casualties", "road_type",
             "speed_limit", "light_conditions", "weather_conditions",
             "road_surface_conditions", "urban_or_rural_area"]]
    df = df.rename(columns={"local_authority_ons_district": "district",
                            "collision_severity": "severity"})
    # "04/03/2023" + "12:10"  ->  one timestamp
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.drop(columns=["date", "time"])
    df["police_force_name"] = df["police_force"].map(POLICE_FORCES).fillna("Other")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df.to_csv(STAGING / "collisions.csv", index=False)
    log(f"staged {len(df):,} collisions")


def stage_vehicles():
    df = pd.read_csv(RAW / f"vehicle_{YEAR}.csv", low_memory=False)
    df = df[["collision_index", "vehicle_reference", "vehicle_type",
             "age_of_driver", "sex_of_driver"]]
    df.loc[df.age_of_driver < 0, "age_of_driver"] = None   # -1 means "unknown"
    df.to_csv(STAGING / "vehicles.csv", index=False)
    log(f"staged {len(df):,} vehicles")


def stage_casualties():
    df = pd.read_csv(RAW / f"casualty_{YEAR}.csv", low_memory=False)
    df = df[["collision_index", "casualty_reference", "casualty_class",
             "casualty_severity", "age_of_casualty", "sex_of_casualty"]]
    df.loc[df.age_of_casualty < 0, "age_of_casualty"] = None
    df.to_csv(STAGING / "casualties.csv", index=False)
    log(f"staged {len(df):,} casualties")


# -------------------------------------------------------------- 3. VALIDATE
def validate_data():
    """Data-quality checks. Bad rows are NOT silently dropped: they are saved
    to data/rejected/ together with the reason."""
    df = pd.read_csv(STAGING / "collisions.csv")
    checks = {
        "duplicate collision_index": df.duplicated("collision_index"),
        "missing date or time": df["datetime"].isna(),
        "coordinates missing or outside the UK": ~(df.latitude.between(49.8, 61.0)
                                                   & df.longitude.between(-8.7, 2.0)),
        "invalid severity": ~df.severity.isin([1, 2, 3]),
    }
    rejected = []
    for reason, failed in checks.items():
        log(f"check '{reason}': {failed.sum()} rows failed")
        rejected.append(df[failed].assign(reject_reason=reason))
    bad = pd.concat([checks[r] for r in checks], axis=1).any(axis=1)
    pd.concat(rejected).to_csv(REJECTED / "rejected_collisions.csv", index=False)
    df[~bad].to_csv(STAGING / "collisions.csv", index=False)

    # vehicles / casualties must belong to a valid collision
    valid_ids = set(df.loc[~bad, "collision_index"])
    for name in ("vehicles", "casualties"):
        child = pd.read_csv(STAGING / f"{name}.csv")
        orphan = ~child.collision_index.isin(valid_ids)
        child[orphan].assign(reject_reason="no matching collision").to_csv(
            REJECTED / f"rejected_{name}.csv", index=False)
        child[~orphan].to_csv(STAGING / f"{name}.csv", index=False)
        log(f"{name}: {orphan.sum()} rows without a valid collision rejected")
    log(f"{bad.sum()} collisions rejected, {(~bad).sum():,} passed")


# ------------------------------------------------------------------ 4. JOIN
def join_accident_weather():
    """Attach the weather at the accident's grid square and nearest hour."""
    acc = pd.read_csv(STAGING / "collisions.csv", parse_dates=["datetime"])
    weather = pd.read_csv(RAW / f"weather_{YEAR}.csv", parse_dates=["time"])
    weather = weather.drop_duplicates(["cell_lat", "cell_lon", "time"])

    acc["cell_lat"] = np.floor(acc.latitude) + 0.5
    acc["cell_lon"] = np.floor(acc.longitude) + 0.5
    # accidents in squares without weather use the nearest square that has it
    squares = weather[["cell_lat", "cell_lon"]].drop_duplicates().to_numpy()
    def nearest(lat, lon):
        distance = (squares[:, 0] - lat) ** 2 + (squares[:, 1] - lon) ** 2
        return squares[distance.argmin()]
    lookup = {(la, lo): nearest(la, lo)
              for la, lo in acc[["cell_lat", "cell_lon"]].drop_duplicates().to_numpy()}
    acc[["cell_lat", "cell_lon"]] = [lookup[(la, lo)] for la, lo in
                                     acc[["cell_lat", "cell_lon"]].to_numpy()]
    acc["time"] = acc["datetime"].dt.round("h")       # nearest hour
    acc = acc.merge(weather, on=["cell_lat", "cell_lon", "time"], how="left")
    acc = acc.rename(columns={"temperature_2m": "temperature_c",
                              "precipitation": "precipitation_mm",
                              "wind_speed_10m": "wind_speed_kmh"})
    acc = acc.drop(columns=["cell_lat", "cell_lon", "time"])
    acc.to_csv(STAGING / "collisions_weather.csv", index=False)
    log(f"weather matched for {acc.temperature_c.notna().mean():.1%} of accidents")


# ------------------------------------------------------------------ 5. LOAD
def load_postgresql():
    """Load the cleaned tables into PostgreSQL (replaced on every run)."""
    db = engine()
    for table, file in [("collisions", "collisions_weather.csv"),
                        ("vehicles", "vehicles.csv"), ("casualties", "casualties.csv")]:
        df = pd.read_csv(STAGING / file)
        with db.begin() as conn:      # drop dependent warehouse tables first
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        df.to_sql(table, db, index=False, chunksize=10_000, method="multi")
        log(f"loaded {len(df):,} rows into table '{table}'")


# ------------------------------------------------------------------ 6. MART
def build_analytical_mart():
    """Build the star schema + summary views (see sql/warehouse.sql)."""
    sql = (ROOT / "sql" / "warehouse.sql").read_text()
    with engine().begin() as conn:
        conn.execute(text(sql))
    log("star schema and analytical views built")


STEPS = [extract_accident_data, extract_weather_data, stage_collisions,
         stage_vehicles, stage_casualties, validate_data,
         join_accident_weather, load_postgresql, build_analytical_mart]

if __name__ == "__main__":
    for step in STEPS:
        log(f"--- {step.__name__}")
        step()

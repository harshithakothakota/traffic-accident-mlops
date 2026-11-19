# UK Traffic Accident Severity – Analytics and Prediction (2023)

A small end-to-end Data Engineering + MLOps project:

```
STATS19 accident CSVs ─┐
                       ├─> extract ─> stage ─> validate ─> join weather ─> PostgreSQL ─> star schema + views ─> Streamlit dashboard
Open-Meteo weather API ┘   (all orchestrated by one Airflow DAG)

PostgreSQL ─> train.py (3 models, tracked in MLflow) ─> best model registered ─> FastAPI /predict ─> dashboard "Predict" page
                                                                               └> monitor.py (drift + missed serious accidents)
```

## How to run

First time only (installs Python packages, PostgreSQL and creates the database):
```bash
./run.sh setup
```

Then:
```bash
./run.sh pipeline   # 1. Airflow runs the ETL pipeline (about 1 min; first run also downloads the data)
./run.sh train      # 2. trains the models, logs them to MLflow, runs monitoring
./run.sh start      # 3. starts the database, API, dashboard and MLflow
./run.sh airflow    # (optional) Airflow web UI on http://localhost:8080
./run.sh stop       # stop everything
```

| What | Where |
|---|---|
| Dashboard | http://localhost:8501 |
| API (try `/predict` here) | http://localhost:8000/docs |
| MLflow (runs + registered model) | http://localhost:5000 |
| Airflow | http://localhost:8080 (user `admin`, password in `airflow_home/standalone_admin_password.txt`) |

The database password lives in `.env` (created by setup). Don't submit `.env`;
`.env.example` shows the format.

## What each file does

| File | Job |
|---|---|
| `etl.py` | the whole data pipeline: 9 small functions, one per step (see below) |
| `dags/accident_pipeline_dag.py` | Airflow DAG that runs those 9 functions in order, daily |
| `sql/warehouse.sql` | builds the star schema (5 dimensions + 1 fact table) and the summary views |
| `dashboard.py` | Streamlit app: KPIs, 7 charts, a top-10 table and the prediction page |
| `train.py` | trains Random Forest, XGBoost and LightGBM, logs results to MLflow, registers the best model |
| `api.py` | FastAPI service with `/health` and `/predict` |
| `monitor.py` | checks the model month by month for drift and missed serious accidents |
| `Dockerfile` | packages the API as a container |
| `run.sh` | shortcuts for the commands above |

### The 9 pipeline steps (`etl.py`)
1. **extract_accident_data** – downloads the 3 STATS19 files for 2023 (collisions, vehicles, casualties) into `data/raw/`.
2. **extract_weather_data** – splits the UK into 1° × 1° squares and downloads hourly 2023 weather (temperature, rain, wind) for each square that has at least 50 accidents (42 squares).
3. **stage_collisions** – keeps the useful columns, joins date + time into one timestamp, and adds the police force name.
4. **stage_vehicles** – keeps the useful vehicle columns; age `-1` (unknown) becomes empty.
5. **stage_casualties** – same idea for casualties.
6. **validate_data** – removes duplicates, missing dates and coordinates outside the UK, plus vehicles and casualties with no matching accident. Rejected rows are **saved with the reason** in `data/rejected/`, not silently deleted.
7. **join_accident_weather** – gives each accident the weather in its grid square at the nearest hour.
8. **load_postgresql** – loads the clean tables into PostgreSQL.
9. **build_analytical_mart** – runs `sql/warehouse.sql`.

### Star schema
`fact_accidents` (one row per accident) is linked to `dim_date`, `dim_location`
(police force, district, urban/rural), `dim_road` (road type, speed limit),
`dim_conditions` (weather, road surface, light) and `dim_severity`.
The view `mart_accidents` joins them all for the dashboard.

## Results (2023)
* 104,258 accidents downloaded. **12 rejected** (bad coordinates), plus 21 vehicles and 13 casualties with no matching accident. **104,246 loaded.**
* Weather matched for **100%** of accidents.
* Dashboard: 104,246 accidents, 132,964 casualties, 1,522 fatal. 23.9% were fatal or serious.
* Models, trained on Jan–Sep and tested on Oct–Dec:

| Model | Accuracy | Macro-F1 | Recall Fatal | Recall Serious | Recall Slight |
|---|---|---|---|---|---|
| **Random Forest (chosen)** | **0.617** | **0.392** | 0.19 | 0.41 | 0.69 |
| XGBoost | 0.599 | 0.387 | 0.22 | 0.44 | 0.65 |
| LightGBM | 0.565 | 0.376 | 0.33 | 0.43 | 0.61 |

* Monitoring (Oct–Dec): no class or location drift (PSI ≤ 0.011). The model calls about 50% of fatal and serious accidents "Slight" (limit 70%), so no retraining is needed.

## Likely questions and short answers
* **Why combine weather?** Weather is a likely contributing factor, and the brief asks for it. We match each accident to the weather at the same place and hour.
* **Why a 1° grid?** The free weather API limits requests. One request per grid square keeps it small (42 requests).
* **What happens to bad data?** It isn't deleted silently. It's saved in `data/rejected/` with the reason (12 accidents had coordinates outside the UK).
* **Why a star schema?** The facts (accidents) sit in the middle and the descriptions (date, place, road, conditions, severity) around them, so dashboard queries are simple GROUP BYs.
* **Why is accuracy only 62%?** Severity mostly depends on things that aren't in the data (impact speed, seat belts). Also, 76% of accidents are Slight, so a model that always says "Slight" would get 76% accuracy but miss every serious crash. That's why we use **class weights** and judge models by **macro-F1** and **recall**, not accuracy.
* **What is macro-F1?** The F1 score for each class (Fatal, Serious, Slight) averaged equally, so the rare classes count as much as the common one.
* **Why train on Jan–Sep and test on Oct–Dec?** It mimics real use: learn from the past, predict the future. A random split would let the model "see the future".
* **What does MLflow do here?** It records each model's settings, scores and confusion matrix, and stores the chosen model as version 1 with the alias `production`.
* **What is the pipeline in `train.py`?** Filling missing values plus the model, saved as one object, so the API applies exactly the same preprocessing.
* **What is PSI?** Population Stability Index, a number for how much a distribution has changed (0 = same, over 0.1 = noticeable).
* **When would you retrain?** If drift goes over 0.1, or the model starts missing more than 70% of serious accidents, or a new year of data is published.

## Data sources
* DfT Road Safety Data (STATS19), 2023: https://www.data.gov.uk/dataset/road-accidents-safety-data (Open Government Licence)
* Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api (CC BY 4.0, no key needed)

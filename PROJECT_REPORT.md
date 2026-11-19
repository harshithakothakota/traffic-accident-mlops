# UK Traffic Accident Severity – Analytics and Prediction (2023)

*Data Engineering and MLOps – Individual Project 4*

> Source material for the project report and presentation. All numbers come
> from the executed project. The figures are in `screenshots/` (listed at the
> end, together with a suggested slide outline).

---

## Abstract
Road accidents cause deaths, injuries and large social costs. This project
builds a simple end-to-end pipeline that combines the UK government's 2023 road
accident records (STATS19) with historical hourly weather from Open-Meteo. An
Apache Airflow DAG extracts, stages, validates and joins the data and loads it
into a PostgreSQL star schema. A Streamlit dashboard shows severity, time,
weather, road-type and location patterns. For the MLOps part, three
classifiers (Random Forest, XGBoost, LightGBM) predict accident severity
(Fatal / Serious / Slight). Experiments are tracked in MLflow, the best model
is registered and served through a FastAPI API (with a Dockerfile), and a
monitoring script checks for drift and missed serious accidents.

**Keywords:** STATS19, Open-Meteo, Apache Airflow, PostgreSQL, star schema, Streamlit, MLflow, FastAPI, Docker, accident severity.

---

## 1. Introduction and objectives
Road-safety teams need to know **where, when and under what conditions**
accidents happen and how severe they are. The raw data comes as separate
files (accidents, vehicles, casualties) and has no weather information. This
project:

1. automatically downloads accident and weather data,
2. cleans and validates it, keeping a log of rejected records,
3. joins each accident to the weather at that place and hour,
4. stores it in a PostgreSQL star schema built for analysis,
5. shows the patterns in an interactive dashboard,
6. trains, tracks, deploys and monitors a model that predicts severity.

---

## 2. Data sources
| Source | Used for | Access |
|---|---|---|
| **DfT Road Safety Data (STATS19), 2023** | 3 CSV files: collisions (104,258 rows), vehicles (189,815), casualties (132,977) | `https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{collision,vehicle,casualty}-2023.csv`. Free, Open Government Licence |
| **Open-Meteo Historical Weather API** | hourly temperature, precipitation and wind speed for 2023 | `https://archive-api.open-meteo.com/v1/archive`. Free, no API key, CC BY 4.0 |

The data is anonymised and contains no personal information. The database
password is kept in a `.env` file, not in the code.

---

## 3. Architecture
Figure: `screenshots/0_architecture.png`

**Part 1:** Data sources → Extract (`data/raw/`) → Stage + validate (rejected
rows saved to `data/rejected/`) → Join weather → PostgreSQL star schema + views
→ Streamlit dashboard. All of it is orchestrated by the Airflow DAG
`accident_etl_pipeline`.

**Part 2:** `train.py` (3 models) → MLflow (metrics + model registry) →
FastAPI `/predict` (Dockerfile) → dashboard prediction page, plus
`monitor.py` (drift + missed serious accidents).

| Layer | Technology |
|---|---|
| Ingestion and processing | Python, Pandas, requests |
| Orchestration | Apache Airflow 2.10 |
| Storage | PostgreSQL 16 |
| Dashboard | Streamlit + Plotly |
| Machine learning | scikit-learn (Random Forest), XGBoost, LightGBM |
| Experiment tracking and registry | MLflow |
| Serving | FastAPI + Uvicorn, Docker |

---

## 4. Data ingestion and ETL pipeline (`etl.py`, Airflow DAG)
The DAG `accident_etl_pipeline` runs daily with 1 retry. It has 9 tasks, each
one a small Python function (figure: `screenshots/7_airflow_dag_graph.png`):

| Task | What it does |
|---|---|
| extract_accident_data | downloads the 3 STATS19 files for 2023 (with retries; skips files already downloaded) |
| extract_weather_data | splits the UK into 1° × 1° squares and downloads hourly 2023 weather for the 42 squares with ≥ 50 accidents (367,920 hourly rows) |
| stage_collisions | keeps 16 useful columns, combines date + time into one timestamp, adds the police-force name |
| stage_vehicles | keeps vehicle type and driver age / sex; `-1` (unknown) becomes empty |
| stage_casualties | keeps casualty class, severity, age and sex |
| validate_data | data-quality checks; rejected rows are written to `data/rejected/` with the reason |
| join_accident_weather | matches each accident to the weather in its grid square at the nearest hour (nearest square if its own has no data) |
| load_postgresql | loads the clean tables into PostgreSQL |
| build_analytical_mart | builds the star schema and summary views (`sql/warehouse.sql`) |

**Data layers:** raw files (`data/raw/`, never changed) → staging files
(`data/staging/`) → cleaned tables in PostgreSQL → star schema and analytical
views.

**Execution:** the full DAG ran successfully, all 9 tasks green, in about
1 minute once the data was downloaded (`screenshots/6_airflow_dag_runs.png`).

---

## 5. Data quality
| Check | Failed rows | Action |
|---|---|---|
| Duplicate accident ID | 0 | reject |
| Missing / unparseable date or time | 0 | reject |
| Coordinates missing or outside the UK | **12** | reject |
| Invalid severity (not 1, 2 or 3) | 0 | reject |
| Vehicles with no matching accident | **21** | reject |
| Casualties with no matching accident | **13** | reject |

Result: **104,246 accidents** loaded (104,258 − 12). Weather was matched for
**100%** of them. Rejected rows are kept with their reason in
`data/rejected/`, not silently deleted.

---

## 6. Database design (star schema)
| Table | Rows | Contents |
|---|---|---|
| `fact_accidents` | 104,246 | one row per accident: date, hour, severity, links to dimensions, vehicles, casualties, temperature, precipitation, wind, coordinates |
| `dim_date` | 365 | date, month, weekday, weekend flag |
| `dim_location` | 693 | police force (name), district, urban / rural |
| `dim_road` | 34 | road type, speed limit |
| `dim_conditions` | 196 | reported weather, road surface, light |
| `dim_severity` | 3 | Fatal, Serious, Slight |

Views for analysis: `mart_accidents` (everything joined, used by the
dashboard), `mart_by_hour`, `mart_by_weekday`, `mart_by_weather`,
`mart_by_road_type`, `mart_top_areas`. Primary and foreign keys are enforced.

---

## 7. Dashboard and findings
Streamlit app (`screenshots/1_dashboard.png`). Filters: severity, road type,
weather, weekday.

**KPIs (2023):** 104,246 accidents · 132,964 casualties · 1,522 fatal accidents · 23.9% fatal or serious.

| View | Finding |
|---|---|
| 1. Severity distribution | 76.1% Slight, 22.5% Serious, 1.46% Fatal: a very imbalanced target |
| 2. Accidents by hour | peak at **17:00** (9,212 accidents); a smaller morning peak around 08:00 |
| 3. Accidents by weekday | **Friday** is the busiest day (17,116), Sunday the quietest |
| 4. Weather conditions | most accidents happen in **fine weather**, simply because most driving happens in fine weather |
| 5. Road type | most accidents are on **single carriageways** |
| 6. Observed temperature (Open-Meteo) | most accidents happen between about 5 and 25 °C |
| 7. Hotspot map | fatal and serious accidents cluster in London, Birmingham, Manchester, Leeds and along major roads |
| Top 10 areas | **Metropolitan Police** (London) has by far the most accidents (22,740), then West Midlands (4,688) and West Yorkshire (4,247) |

Extra findings from the database:
* **Rural roads are more dangerous:** 29.1% of rural accidents are fatal or serious, versus 21.4% urban, and the fatal rate is 2.81% versus 0.81% (3.5×).
* **Speed matters:** the fatal-or-serious share rises from 18.4% at 20 mph to 34.1% at 60 mph.
* **Darkness:** 26.5% of accidents in darkness are fatal or serious, versus 22.9% in daylight.

---

## 8. Machine learning and MLOps
**Goal:** predict an accident's severity (Fatal / Serious / Slight) from
information known when it is reported.

**Features (15):** hour, weekday, month, latitude, longitude, urban/rural, road
type, speed limit, light, weather, road surface, number of vehicles,
temperature, precipitation, wind speed.

**Method**
* **Chronological split:** train on Jan–Sep 2023 (77,657 accidents), test on Oct–Dec 2023 (26,589). Learning from the past and testing on the future avoids leaking future information.
* **Class imbalance:** only 1.5% of accidents are fatal, so **balanced class weights** make mistakes on rare classes count more.
* **One inference pipeline:** missing-value filling + model, saved as one object, so the API uses exactly the same preprocessing.
* **Metrics:** accuracy, **macro-F1** (treats the three classes equally) and recall per class, plus a confusion matrix for each model.

**Results (test set, Oct–Dec 2023)**

| Model | Accuracy | Macro-F1 | Recall Fatal | Recall Serious | Recall Slight |
|---|---|---|---|---|---|
| **Random Forest (selected)** | **0.617** | **0.392** | 0.19 | 0.41 | 0.69 |
| XGBoost | 0.599 | 0.387 | 0.22 | 0.44 | 0.65 |
| LightGBM | 0.565 | 0.376 | 0.33 | 0.43 | 0.61 |

Confusion matrices: `models/confusion_matrix_*.png`.

**Interpretation:** accuracy looks modest, but a model that always says
"Slight" would score about 76% accuracy while missing every fatal and serious
accident. Our models trade some accuracy for finding the rare, important
classes; for example, LightGBM finds 33% of fatal accidents. Severity is hard
to predict from this data because the biggest factors (impact speed, seat-belt
use, vehicle safety) are not recorded.

**MLflow:** experiment `accident_severity_classification` holds the 3 runs
with their parameters, metrics and confusion matrices
(`screenshots/4_mlflow_runs.png`). The best model (highest macro-F1) is
registered as **`accident_severity_classifier` version 1** with the alias
**`production`** (`screenshots/5_mlflow_registered_model.png`).

**Deployment:** FastAPI app "Accident Severity Prediction API" with `GET /health`
and `POST /predict`. Inputs are validated: for example, hour must be 0–23 and
coordinates must be inside the UK (`screenshots/3_fastapi_docs.png`). A
`Dockerfile` packages the API. The dashboard's **Predict Severity** page sends
the form to the API and shows the predicted class and probabilities
(`screenshots/2_prediction_page.png`). Example: 17:00 Friday in July, London,
urban 30 mph single carriageway, daylight, dry → **Serious** (48%) vs Slight
(44%). Class weighting makes the model cautious.

**Monitoring (`monitor.py`)**, one row per test month:

| Month | Accidents | Class drift (PSI) | Location drift (PSI) | Serious/fatal predicted as Slight | Retrain? |
|---|---|---|---|---|---|
| Oct | 9,142 | 0.000 | 0.007 | 51.0% | No |
| Nov | 9,160 | 0.000 | 0.010 | 50.2% | No |
| Dec | 8,287 | 0.001 | 0.011 | 49.6% | No |

Limits: drift PSI > 0.10, or more than 70% of serious and fatal accidents
missed. **Retraining criteria:** retrain when a limit is crossed or a new
year of STATS19 data is published. Rerun `train.py`; the new model becomes the
next registered version, and the `production` alias moves to it if it scores
better.

---

## 9. Conclusion and future work
The project delivers a working, repeatable pipeline. Airflow loads 2023 UK
accident and weather data into a PostgreSQL star schema with logged data-quality
checks. A Streamlit dashboard shows clear patterns: evening rush-hour peaks,
Fridays, London as the hotspot, and rural, high-speed and dark roads being
more dangerous. In Part 2, the severity model is tracked and registered in
MLflow, served through FastAPI and monitored for drift.

**Limitations:** one year of data, weather at a coarse 1° grid, modest model
performance (severity depends on factors not recorded), and the Dockerfile is
provided but was not built on the development machine.

**Future work:** more years of data, finer weather resolution, OpenStreetMap
road details, model explanations (SHAP), automatic retraining in Airflow, and
cloud deployment.

---

## References
1. Department for Transport, Road Safety Data (STATS19), data.gov.uk.
2. Open-Meteo, Historical Weather API, open-meteo.com.
3. Apache Airflow documentation, airflow.apache.org.
4. PostgreSQL documentation, postgresql.org.
5. Streamlit documentation, docs.streamlit.io.
6. MLflow documentation, mlflow.org.
7. FastAPI documentation, fastapi.tiangolo.com.
8. scikit-learn, XGBoost and LightGBM documentation.

---

## Figures (in `screenshots/` unless noted)
| File | Shows |
|---|---|
| `0_architecture.png` | architecture diagram |
| `1_dashboard.png` | full dashboard: KPIs, 7 views, top-10 table |
| `2_prediction_page.png` | prediction page with result |
| `3_fastapi_docs.png` | FastAPI documentation page |
| `4_mlflow_runs.png` | MLflow experiment with the 3 model runs |
| `5_mlflow_registered_model.png` | registered model, version 1, alias production |
| `6_airflow_dag_runs.png` | Airflow DAG run, all tasks green |
| `7_airflow_dag_graph.png` | Airflow DAG graph of the 9 tasks |
| `models/confusion_matrix_random_forest.png` (and xgboost, lightgbm) | confusion matrices |

## Suggested slides (about 12)
1. Title
2. Problem and objectives
3. Data sources (STATS19 + Open-Meteo)
4. Architecture (`0_architecture.png`)
5. Airflow pipeline: 9 tasks (`7_airflow_dag_graph.png`)
6. Data quality: checks table, 12 / 21 / 13 rejected, 100% weather match
7. Star schema: table of dimensions and fact
8. Dashboard (`1_dashboard.png`): KPIs and key findings
9. Model training and results table + confusion matrix
10. MLflow registry + FastAPI + prediction page
11. Monitoring and retraining rules
12. Conclusion, limitations, future work

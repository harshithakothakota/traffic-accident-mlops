"""Simple model monitoring: compares each test month with the training data.

    python monitor.py

Checks
  * class drift     - has the mix of Fatal / Serious / Slight accidents changed?
  * location drift  - has the share of accidents per police force changed?
  * false negatives - how many fatal/serious accidents does the model call "Slight"?
Retrain if any check goes past its limit.
"""
import joblib
import numpy as np
import pandas as pd

from etl import ROOT, engine
from train import FEATURES

LIMITS = {"class_drift_psi": 0.10, "location_drift_psi": 0.10,
          "false_negative_rate": 0.70}


def psi(expected, actual):
    """Population Stability Index: 0 = identical, > 0.1 = noticeable change."""
    e, a = expected.align(actual, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def main():
    df = pd.read_sql("""
        SELECT severity - 1 AS label, police_force,
               extract(hour   FROM datetime::timestamp) AS hour,
               extract(isodow FROM datetime::timestamp) AS weekday_number,
               extract(month  FROM datetime::timestamp) AS month,
               latitude, longitude, urban_or_rural_area, road_type, speed_limit,
               light_conditions, weather_conditions, road_surface_conditions,
               number_of_vehicles, temperature_c, precipitation_mm, wind_speed_kmh
        FROM collisions""", engine())
    model = joblib.load(ROOT / "models" / "model.joblib")
    train = df[df.month <= 9]

    rows = []
    for month, part in df[df.month >= 10].groupby("month"):
        pred = model.predict(part[FEATURES])
        severe = part.label <= 1                      # fatal or serious
        rows.append({
            "month": int(month),
            "accidents": len(part),
            "class_drift_psi": psi(train.label.value_counts(normalize=True),
                                   part.label.value_counts(normalize=True)),
            "location_drift_psi": psi(train.police_force.value_counts(normalize=True),
                                      part.police_force.value_counts(normalize=True)),
            "false_negative_rate": float(((pred == 2) & severe).sum() / severe.sum()),
        })
    report = pd.DataFrame(rows).round(3)
    report["retrain_needed"] = (report[list(LIMITS)] > pd.Series(LIMITS)).any(axis=1)
    report.to_csv(ROOT / "models" / "monitoring_report.csv", index=False)
    print(report.to_string(index=False))
    print("\nlimits:", LIMITS)
    print("RETRAIN RECOMMENDED" if report.retrain_needed.any() else "Model OK - no retraining needed")


if __name__ == "__main__":
    main()

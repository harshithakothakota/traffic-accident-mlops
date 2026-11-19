"""Train severity models, track them in MLflow, register the best one.

    python train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db      (to view the results)
"""
import json
import logging
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from etl import ROOT, engine

logging.getLogger("mlflow").setLevel(logging.ERROR)   # hide MLflow info/warning noise

FEATURES = ["hour", "weekday_number", "month", "latitude", "longitude",
            "urban_or_rural_area", "road_type", "speed_limit", "light_conditions",
            "weather_conditions", "road_surface_conditions", "number_of_vehicles",
            "temperature_c", "precipitation_mm", "wind_speed_kmh"]
LABELS = ["Fatal", "Serious", "Slight"]          # class 0, 1, 2
MODEL_NAME = "accident_severity_classifier"


def load_data():
    df = pd.read_sql("""
        SELECT severity - 1 AS label,                       -- 0 Fatal, 1 Serious, 2 Slight
               extract(hour   FROM datetime::timestamp) AS hour,
               extract(isodow FROM datetime::timestamp) AS weekday_number,
               extract(month  FROM datetime::timestamp) AS month,
               latitude, longitude, urban_or_rural_area, road_type, speed_limit,
               light_conditions, weather_conditions, road_surface_conditions,
               number_of_vehicles, temperature_c, precipitation_mm, wind_speed_kmh
        FROM collisions""", engine())
    # chronological split: learn from Jan-Sep, test on the unseen Oct-Dec
    train, test = df[df.month <= 9], df[df.month >= 10]
    return train[FEATURES], train.label, test[FEATURES], test.label


MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                            class_weight="balanced", n_jobs=-1,
                                            random_state=42),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=42),
    "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", random_state=42, verbose=-1),
}


def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"train rows: {len(X_train):,}   test rows: {len(X_test):,}")
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("accident_severity_classification")

    results = []
    for name, model in MODELS.items():
        # preprocessing (fill missing values) + model = ONE pipeline object
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("model", model)])
        # class imbalance: only ~1.5% of accidents are fatal, so rare classes get
        # a higher weight ("balanced" weights; XGBoost takes them as sample weights)
        weights = compute_sample_weight("balanced", y_train)
        with mlflow.start_run(run_name=name) as run:
            if name == "xgboost":
                pipeline.fit(X_train, y_train, model__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            recall = recall_score(y_test, pred, average=None)
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "recall_fatal": recall[0],
                "recall_serious": recall[1],
                "recall_slight": recall[2],
            }
            mlflow.log_params({"model": name, "class_imbalance": "balanced weights",
                               "train_period": "Jan-Sep 2023", "test_period": "Oct-Dec 2023"})
            mlflow.log_metrics(metrics)

            ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=LABELS,
                                                    normalize="true", values_format=".0%")
            plt.title(f"{name} - confusion matrix (test)")
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.savefig(ROOT / "models" / f"confusion_matrix_{name}.png", dpi=120)
            plt.close()

            mlflow.sklearn.log_model(
                pipeline, name="model", input_example=X_test.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            results.append({"name": name, "run_id": run.info.run_id,
                            "pipeline": pipeline, **metrics})
        print(f"{name:14s} accuracy={metrics['accuracy']:.3f}  macro-F1={metrics['macro_f1']:.3f}  "
              f"recall fatal={recall[0]:.2f} serious={recall[1]:.2f} slight={recall[2]:.2f}")

    # best model = highest macro-F1 (treats the 3 classes equally)
    best = max(results, key=lambda r: r["macro_f1"])
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)
    print(f"\nbest model: {best['name']} -> registered as {MODEL_NAME} "
          f"version {version.version} (alias 'production')")

    # copy for the API (so the API does not need the MLflow server)
    joblib.dump(best["pipeline"], ROOT / "models" / "model.joblib")
    info = {"model": best["name"], "version": version.version, "features": FEATURES,
            "labels": LABELS, **{k: round(best[k], 4) for k in
                                 ("accuracy", "macro_f1", "recall_fatal",
                                  "recall_serious", "recall_slight")}}
    (ROOT / "models" / "model_info.json").write_text(json.dumps(info, indent=2))
    pd.DataFrame(results).drop(columns="pipeline").to_csv(
        ROOT / "models" / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()

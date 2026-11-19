"""FastAPI prediction service.

    uvicorn api:app --port 8000          then open http://localhost:8000/docs
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS = Path(__file__).resolve().parent / "models"
model = joblib.load(MODELS / "model.joblib")            # preprocessing + model pipeline
info = json.loads((MODELS / "model_info.json").read_text())

app = FastAPI(title="Accident Severity Prediction API", version="1.0.0")


class Accident(BaseModel):
    """Details of one accident (STATS19 codes)."""
    hour: int = Field(..., ge=0, le=23)
    weekday_number: int = Field(..., ge=1, le=7, description="1 = Monday ... 7 = Sunday")
    month: int = Field(..., ge=1, le=12)
    latitude: float = Field(..., ge=49.8, le=61.0)
    longitude: float = Field(..., ge=-8.7, le=2.0)
    urban_or_rural_area: int = Field(1, description="1 = Urban, 2 = Rural")
    road_type: int = Field(6, description="1 Roundabout, 2 One way, 3 Dual, 6 Single, 7 Slip road")
    speed_limit: int = 30
    light_conditions: int = Field(1, description="1 = Daylight, 4-7 = Darkness")
    weather_conditions: int = Field(1, description="1 Fine, 2 Raining, 3 Snowing, 7 Fog")
    road_surface_conditions: int = Field(1, description="1 Dry, 2 Wet, 4 Frost/ice")
    number_of_vehicles: int = Field(1, ge=1)
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    wind_speed_kmh: float | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": info["model"], "model_version": info["version"]}


@app.post("/predict")
def predict(accident: Accident):
    row = pd.DataFrame([accident.model_dump()])[info["features"]]
    probabilities = model.predict_proba(row)[0]
    predicted = int(probabilities.argmax())
    return {
        "predicted_severity": info["labels"][predicted],
        "probabilities": {label: round(float(p), 3)
                          for label, p in zip(info["labels"], probabilities)},
        "model_version": info["version"],
    }

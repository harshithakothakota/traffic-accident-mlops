# Container for the prediction API.
#   docker build -t accident-api .
#   docker run -p 8000:8000 accident-api      -> http://localhost:8000/docs
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas==3.1.1 scikit-learn==1.9.1 xgboost==3.4.1 \
    lightgbm==4.7.0 joblib==1.6.0 fastapi==0.163.1 uvicorn==0.54.0
COPY api.py .
COPY models/model.joblib models/model_info.json models/
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

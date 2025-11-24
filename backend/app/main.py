from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    HousingFeatures,
    HousingPredictionResponse,
    ElectricityFeatures,
    ElectricityPredictionResponse,
    HealthResponse,
)
from .services.model_service import (
    predict_housing,
    predict_electricity,
    check_models_available,
)

app = FastAPI(
    title="Cloud-Serfers ML Backend",
    description="Backend API for UK Housing & Electricity demand predictions",
    version="1.0.0",
)

# Allow frontend to call backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    housing_ok, electricity_ok = check_models_available()
    if housing_ok and electricity_ok:
        print("✅ Both housing and electricity models loaded successfully.")
    else:
        print("⚠️ Could not load one or both models on startup.")

# ✅ HOME PAGE so teacher sees something
@app.get("/")
def root():
    return {
        "message": "API is running ✅",
        "docs": "/docs",
        "health": "/health",
        "predict_housing": "/predict/housing (POST)",
        "predict_electricity": "/predict/electricity (POST)"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    housing_ok, electricity_ok = check_models_available()
    ok = housing_ok and electricity_ok
    status = "ok" if ok else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=ok,
        housing_model_available=housing_ok,
        electricity_model_available=electricity_ok,
    )

@app.post("/predict/housing", response_model=HousingPredictionResponse)
def predict_housing_endpoint(features: HousingFeatures):
    try:
        prediction = predict_housing(features.dict())
        return HousingPredictionResponse(predicted_price=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Housing prediction error: {e}")

@app.post("/predict/electricity", response_model=ElectricityPredictionResponse)
def predict_electricity_endpoint(features: ElectricityFeatures):
    try:
        prediction = predict_electricity(features.dict())
        return ElectricityPredictionResponse(predicted_demand=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Electricity prediction error: {e}")

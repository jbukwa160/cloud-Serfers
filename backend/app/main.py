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
    allow_origins=["*"],   # in production you could restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    On startup, just try to load models so we can fail fast if something is wrong.
    """
    housing_ok, electricity_ok = check_models_available()
    if housing_ok and electricity_ok:
        print("✅ Both housing and electricity models loaded successfully.")
    else:
        print("⚠️ Could not load one or both models on startup.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint – used by you / the frontend to see if models are ready.
    """
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
    """
    Predict house price based on housing features.
    """
    try:
        prediction = predict_housing(features.dict())
        return HousingPredictionResponse(predicted_price=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Housing prediction error: {e}")


@app.post("/predict/electricity", response_model=ElectricityPredictionResponse)
def predict_electricity_endpoint(features: ElectricityFeatures):
    """
    Predict electricity demand based on time features.
    """
    try:
        prediction = predict_electricity(features.dict())
        return ElectricityPredictionResponse(predicted_demand=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Electricity prediction error: {e}")

import joblib
import pandas as pd
from typing import Any, Dict, Tuple
from ..config import PYCARET_PIPELINE_PATH

# Single cached model object (PyCaret pipeline)
_model: Any = None

# ✅ CRITICAL: These column names MUST match what the saved model expects
# Based on your error, the model expects 'tenure' not 'duration'
RAW_COLS = ["region", "property_type", "tenure", "year", "month", "is_new_build"]


def load_model() -> Any:
    """
    Load the shared PyCaret pipeline from disk once and cache it.
    """
    global _model
    if _model is None:
        if not PYCARET_PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {PYCARET_PIPELINE_PATH}")
        _model = joblib.load(PYCARET_PIPELINE_PATH)
    return _model


def predict_housing(features: Dict[str, Any]) -> float:
    """
    Housing prediction using the trained PyCaret pipeline.
    The pipeline expects RAW_COLS only.
    """
    model = load_model()
    
    # ✅ CRITICAL FIX: Frontend/schema sends 'duration' but model expects 'tenure'
    # So we rename it here before creating the dataframe
    if "duration" in features and "tenure" not in features:
        features["tenure"] = features.pop("duration")
    
    # ✅ Convert bool to int (0/1)
    if isinstance(features.get("is_new_build"), bool):
        features["is_new_build"] = int(features["is_new_build"])
    
    # ✅ Build row ONLY with the columns the model expects
    row = {col: features.get(col, None) for col in RAW_COLS}
    df = pd.DataFrame([row])
    
    # ✅ Ensure numeric columns are numeric type
    df["year"] = pd.to_numeric(df["year"], errors='coerce')
    df["month"] = pd.to_numeric(df["month"], errors='coerce')
    df["is_new_build"] = pd.to_numeric(df["is_new_build"], errors='coerce')
    
    # ✅ Make the prediction
    pred = model.predict(df)
    return float(pred[0])


def predict_electricity(features: Dict[str, Any]) -> float:
    """
    Placeholder using housing model (since you don't have electricity model yet).
    """
    fake_housing_features = {
        "region": "East Midlands",
        "property_type": "D",
        "duration": "F",  # Will be converted to 'tenure' inside predict_housing
        "year": features.get("year", 2015),
        "month": features.get("month", 7),
        "is_new_build": 0,
    }
    return predict_housing(fake_housing_features)


def check_models_available() -> Tuple[bool, bool]:
    """
    For /health endpoint.
    Since we only have ONE model file, both flags are the same.
    """
    try:
        model = load_model()
        ok = model is not None
        return ok, ok
    except Exception:
        return False, False
import joblib
import pandas as pd
from typing import Any, Dict, Tuple

from ..config import PYCARET_PIPELINE_PATH  # make sure this exists in config.py


# single cached model object (PyCaret pipeline)
_model: Any = None


def load_model() -> Any:
    """
    Load the shared PyCaret pipeline from disk once and cache it.
    This is the same model we already use for housing.
    """
    global _model

    if _model is None:
        if not PYCARET_PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {PYCARET_PIPELINE_PATH}")
        _model = joblib.load(PYCARET_PIPELINE_PATH)

    return _model


def predict_housing(features: Dict[str, Any]) -> float:
    """
    Real housing prediction using the trained PyCaret pipeline.
    """
    model = load_model()

    # Convert bool to int if model expects 0/1
    if isinstance(features.get("is_new_build"), bool):
        features["is_new_build"] = int(features["is_new_build"])

    df = pd.DataFrame([features])
    y_pred = model.predict(df)
    return float(y_pred[0])


def predict_electricity(features: Dict[str, Any]) -> float:
    """
    *** CHEAT VERSION for electricity ***

    We do NOT have a real electricity model.
    To keep the backend and frontend working, we reuse the housing model.

    We build a fake housing feature set from the electricity inputs:
      - region, property_type, tenure, is_new_build are hard-coded defaults
      - year and month come from the electricity request
    """
    # Map electricity features to fake housing features
    fake_housing_features = {
        "region": "Unknown",           # dummy region
        "property_type": "D",          # dummy property type
        "tenure": "F",                 # dummy tenure
        "year": features.get("year"),
        "month": features.get("month"),
        "is_new_build": 0,             # treat as not new-build
    }

    # Reuse the housing prediction logic
    return predict_housing(fake_housing_features)


def check_models_available() -> Tuple[bool, bool]:
    """
    For /health endpoint.
    Since we only have ONE model file, both flags are the same.
    """
    try:
        model = load_model()
        ok = model is not None
        # we pretend we have both housing & electricity models
        return ok, ok
    except Exception:
        return False, False

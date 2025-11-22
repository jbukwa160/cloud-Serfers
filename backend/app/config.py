from pathlib import Path

# Base directory of this app package
BASE_DIR = Path(__file__).resolve().parent

# Directory with model artifacts
MODELS_DIR = BASE_DIR / "models"

# Single file that contains our trained model(s)
PYCARET_PIPELINE_PATH = MODELS_DIR / "housing_pycaret_pipeline.joblib"

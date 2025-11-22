from pydantic import BaseModel, Field
from typing import Literal


# ======== Housing ========

class HousingFeatures(BaseModel):
    region: str = Field(..., description="Region of the property")
    property_type: str = Field(
        ...,
        description="Property type, e.g. D/S/T/F/O"
    )
    duration: str = Field(
        ..., 
        description="Duration, e.g. F (Freehold), L (Leasehold), U (Unknown)", 
    )
    year: int = Field(..., ge=1995, le=2050, description="Year of sale")
    month: int = Field(..., ge=1, le=12, description="Month of sale (1–12)")
    is_new_build: bool = Field(
        ...,
        description="True if property is a new build"
    )


class HousingPredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted sale price in GBP")


# ======== Electricity ========

class ElectricityFeatures(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    is_weekend: Literal[0, 1] = Field(
        ...,
        description="1 if Saturday/Sunday, 0 for weekdays"
    )


class ElectricityPredictionResponse(BaseModel):
    predicted_demand: float = Field(..., description="Predicted demand in MW")


# ======== Health ========

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    housing_model_available: bool
    electricity_model_available: bool

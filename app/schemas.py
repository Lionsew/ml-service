from pydantic import BaseModel, Field

class ModelInput(BaseModel):
    feature_numeric_1: float = Field(..., description="Primer parámetro numérico")
    feature_numeric_2: float = Field(..., description="Segundo parámetro numérico")
    feature_categorical_1: str = Field(..., description="Categoría asignada")

class ModelOutput(BaseModel):
    prediction: int
    confidence: float
    model_version: str = "v1.0.0"
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
import joblib
from app.schemas import ModelInput, ModelOutput

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_resources["model"] = joblib.load("model_artifacts/model.joblib")
        print("Modelo cargado exitosamente en memoria.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        ml_resources["model"] = None
    yield
    ml_resources.clear()

app = FastAPI(title="ML Production Service", lifespan=lifespan)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if ml_resources.get("model") is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"status": "healthy"}

@app.post("/predict", response_model=ModelOutput)
def predict(payload: ModelInput):
    model = ml_resources.get("model")
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")

    input_data = [[payload.feature_numeric_1, payload.feature_numeric_2]]

    pred = model.predict(input_data)[0]
    prob = getattr(model, "predict_proba", lambda x: [[0.0, 1.0]])(input_data)[0][1]

    return ModelOutput(prediction=int(pred), confidence=float(prob))
from fastapi import APIRouter
from app.service.predict import predict

router = APIRouter()

@router.post("/predict")
def run_prediction(payload: dict):

    text = payload.get("text")
    result = predict(text)

    return {
        "input": text,
        "prediction": result
    }
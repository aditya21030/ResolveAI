from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import predict_intent


app = FastAPI(
    title="ResolveAI API",
    description="AI-powered customer support ticket classification API",
    version="1.0.0"
)


class TicketRequest(BaseModel):
    ticket: str = Field(
        ...,
        min_length=3,
        max_length=1000
    )


@app.get("/")
def home():
    return {
        "message": "ResolveAI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "distilbert_intent_v5"
    }


@app.post("/predict")
def predict(request: TicketRequest):
    try:
        result = predict_intent(request.ticket)

        if result["confidence"] < 0.75:
            result["action"] = "human_review"
        else:
            result["action"] = "auto_route"

        return result

    except Exception as error:
        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
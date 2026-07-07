import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "ajain2103/resolveai-distilbert-v5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("V5 model loaded successfully!")


def predict_intent(ticket):
    inputs = tokenizer(
        ticket,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    confidence, predicted_id = torch.max(
        probabilities,
        dim=1
    )

    intent = model.config.id2label[predicted_id.item()]

    return {
        "ticket": ticket,
        "intent": intent,
        "confidence": round(confidence.item(), 4)
    }
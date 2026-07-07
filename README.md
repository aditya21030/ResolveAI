# ResolveAI

**AI-powered customer support ticket classification and routing system**

ResolveAI is an end-to-end NLP application that automatically understands customer support messages, classifies them into one of 27 support intents, and decides whether a ticket should be automatically routed or sent for human review based on model confidence.

## Live Demo

**Try ResolveAI:** https://resolveai-aditya21030.streamlit.app

## Problem

Customer support teams receive large numbers of messages such as:

> "My refund was approved but the money has not arrived."

> "I forgot my password and cannot log in."

> "My card keeps getting declined."

Manually reading and routing every ticket is slow and inefficient.

ResolveAI automates this process.

```text
Customer Ticket
       ↓
Fine-tuned DistilBERT
       ↓
Intent Classification
       ↓
Confidence Score
       ↓
┌────────────────────┐
│ Confidence ≥ 75%?  │
└────────────────────┘
       ↓         ↓
      Yes        No
       ↓         ↓
 Auto Route   Human Review
```

## Features

- Classifies customer tickets into 27 support intents
- Fine-tuned DistilBERT model
- Confidence score for every prediction
- Automatic routing for high-confidence predictions
- Human review fallback for uncertain predictions
- FastAPI REST API
- Streamlit web interface
- Input validation and API error handling
- Health-check endpoint
- Model hosted on Hugging Face
- Public deployment on Streamlit Community Cloud

## Example

### Input

```text
I returned my order two weeks ago and received confirmation that my refund was processed, but the money still hasn't appeared in my bank account.
```

### Output

```json
{
  "intent": "track_refund",
  "confidence": 0.997,
  "action": "auto_route"
}
```

For an ambiguous ticket:

```text
I need some help with something related to my account and order, but I'm not sure what to do.
```

ResolveAI produced:

```text
Predicted Intent: Place Order
Confidence: 59.59%
Routing Action: Human Review
```

## Supported Intents

ResolveAI classifies tickets into 27 categories:

- Cancel Order
- Change Order
- Change Shipping Address
- Check Cancellation Fee
- Check Invoice
- Check Payment Methods
- Check Refund Policy
- Complaint
- Contact Customer Service
- Contact Human Agent
- Create Account
- Delete Account
- Delivery Options
- Delivery Period
- Edit Account
- Get Invoice
- Get Refund
- Newsletter Subscription
- Payment Issue
- Place Order
- Recover Password
- Registration Problems
- Review
- Set Up Shipping Address
- Switch Account
- Track Order
- Track Refund

## Model Development

The project was developed through multiple experiments instead of relying on a single model.

| Version | Model | Real-World Accuracy | Benchmark |
|---|---|---:|---|
| V1 | TF-IDF | 62.96% | 27-ticket benchmark |
| V2 | Improved TF-IDF | 62.96% | 27-ticket benchmark |
| V3 | DistilBERT | 66.67% | 27-ticket benchmark |
| V4 | Augmented DistilBERT | 70.37% | 27-ticket benchmark |
| V5 | Corrective DistilBERT | 70.77% | Clean 260-ticket benchmark |
| V6 | Contrastive DistilBERT | 66.15% | Clean 260-ticket benchmark |

> Note: V1–V4 and V5–V6 were evaluated on different benchmark stages, so their accuracy values should not be treated as a direct comparison across all six versions.

## Why V5 Was Selected

V6 achieved excellent validation performance:

```text
Validation Accuracy: 99.72%
Validation F1:       99.72%
```

However, its performance on the cleaned real-world benchmark dropped:

```text
V5: 70.77% (184/260)
V6: 66.15% (172/260)
```

A detailed comparison showed:

```text
Fixed by V6:  14
Broken by V6: 26
Net Change:  -12
```

The contrastive augmentation used for V6 improved some targeted class boundaries but overcorrected others and reduced real-world generalization.

Therefore, V5 was selected as the final production model.

This experiment demonstrates an important machine learning lesson:

**Higher validation accuracy does not always mean better real-world performance.**

## Benchmark Auditing

An initial 270-ticket benchmark was manually audited.

Audit results:

```text
KEEP:     217
RELABEL:   52
REMOVE:     1
```

After further manual review:

```text
Original Benchmark: 270
Manually Relabeled:  27
Removed:              10
Clean Benchmark:     260
```

The cleaned benchmark was used for the final V5 and V6 evaluation.

## System Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
Inference Pipeline
  ↓
Fine-tuned DistilBERT V5
  ↓
Intent + Confidence
  ↓
Confidence Threshold
  ↓
Auto Route / Human Review
```

The project also includes a FastAPI backend for API-based integration.

```text
Client Application
       ↓
POST /predict
       ↓
FastAPI
       ↓
DistilBERT V5
       ↓
Intent + Confidence + Action
```

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "distilbert_intent_v5"
}
```

### Predict Ticket Intent

```http
POST /predict
```

Request:

```json
{
  "ticket": "My refund was approved but the money has not arrived"
}
```

Response:

```json
{
  "ticket": "My refund was approved but the money has not arrived",
  "intent": "track_refund",
  "confidence": 0.993,
  "action": "auto_route"
}
```

## Confidence-Based Routing

ResolveAI uses a configurable confidence threshold.

```text
Confidence ≥ 75% → Auto Route
Confidence < 75% → Human Review
```

The current 75% threshold is an MVP heuristic and can be optimized using production data and business requirements.

## Tech Stack

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- Pandas

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit

### Deployment

- GitHub
- Hugging Face Hub
- Streamlit Community Cloud

## Project Structure

```text
ResolveAI/
│
├── data/
│   └── real_world_test.csv
│
├── models/
│   └── distilbert_intent_v5/
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── notebooks/
│   └── exploration.ipynb
│
├── app.py
├── frontend.py
├── inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aditya21030/ResolveAI.git
cd ResolveAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run frontend.py
```

The model is downloaded automatically from Hugging Face when the application starts.

## Run the FastAPI Backend

Start the API:

```bash
uvicorn app:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## Model Hosting

The fine-tuned V5 model weights are hosted on Hugging Face because the `model.safetensors` file is larger than GitHub's standard per-file limit.

Model repository:

https://huggingface.co/ajain2103/resolveai-distilbert-v5

## Future Improvements

- Add real support-team ticket queues
- Store tickets in a database
- Build an admin dashboard
- Add ticket status tracking
- Optimize the confidence threshold using production data
- Add model monitoring and drift detection
- Deploy the FastAPI backend as a public API
- Add authentication and role-based access
- Track routing accuracy from human feedback

## Author

**Aditya Jain**

GitHub: https://github.com/aditya21030

---

If you found this project useful, consider giving the repository a star.
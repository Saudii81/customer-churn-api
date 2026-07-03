from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn, based on account tenure and charges.",
    version="1.0.0",
)

model = joblib.load("customer_churn_model.pkl")

CHURN_LABELS = {0: "Stay", 1: "Churn"}


# ---------------------------------------------------------------
# Input schema — this is what makes /docs interactive.
# FastAPI auto-generates a fillable form + validation from this.
# ---------------------------------------------------------------
class CustomerData(BaseModel):
    tenure: float = Field(..., ge=0, description="Number of months the customer has been with the company", examples=[12])
    MonthlyCharges: float = Field(..., ge=0, description="Customer's monthly bill amount", examples=[70.35])
    TotalCharges: float = Field(..., ge=0, description="Total amount charged to the customer so far", examples=[840.5])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 840.50}
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float | None = None
    input: CustomerData


# ---------------------------------------------------------------
# Root — a live, in-browser demo form (no Swagger/Postman needed).
# Great for showing off the API on a projector during a talk.
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 480px; margin: 60px auto; color: #202124; }
            h1 { font-size: 22px; }
            label { display: block; margin-top: 14px; font-size: 13px; color: #5F6368; }
            input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
            button { margin-top: 20px; padding: 10px 18px; background: #4285F4; color: white;
                     border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
            button:hover { background: #3367D6; }
            #result { margin-top: 22px; padding: 14px; border-radius: 8px; display: none; font-size: 15px; }
            .stay { background: #E9F7EE; color: #1D6B37; }
            .churn { background: #FCEEEC; color: #8A2E22; }
            a { color: #4285F4; }
        </style>
    </head>
    <body>
        <h1>🔮 Customer Churn Prediction</h1>
        <p style="color:#5F6368; font-size:13px;">Try a live prediction, or explore the full API at <a href="/docs">/docs</a>.</p>

        <label>Tenure (months)</label>
        <input type="number" id="tenure" value="12" min="0">

        <label>Monthly Charges ($)</label>
        <input type="number" id="MonthlyCharges" value="70.35" step="0.01" min="0">

        <label>Total Charges ($)</label>
        <input type="number" id="TotalCharges" value="840.50" step="0.01" min="0">

        <button onclick="predict()">Predict</button>

        <div id="result"></div>

        <script>
            async function predict() {
                const payload = {
                    tenure: parseFloat(document.getElementById("tenure").value),
                    MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
                    TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
                };
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const box = document.getElementById("result");
                if (!res.ok) {
                    box.className = "churn";
                    box.style.display = "block";
                    box.innerText = "Error: could not get a prediction. Check your inputs.";
                    return;
                }
                const data = await res.json();
                box.className = data.label === "Churn" ? "churn" : "stay";
                box.style.display = "block";
                box.innerHTML = `<strong>Prediction: ${data.label}</strong>` +
                    (data.confidence !== null ? `<br>Confidence: ${(data.confidence * 100).toFixed(1)}%` : "");
            }
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------
# Predict — validated input, human-readable + probability output.
# ---------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    features = [[data.tenure, data.MonthlyCharges, data.TotalCharges]]

    try:
        prediction = int(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0][prediction])

    return PredictionResponse(
        prediction=prediction,
        label=CHURN_LABELS.get(prediction, "Unknown"),
        confidence=confidence,
        input=data,
    )

from fastapi import FastAPI
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("customer_churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    prediction = model.predict([[
        data["tenure"],
        data["MonthlyCharges"],
        data["TotalCharges"]
    ]])

    return {
        "prediction": prediction.tolist()
    }
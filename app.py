import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

MODEL_PATH = "fraud_model.pkl"

app = Flask(__name__, template_folder="templates")

# -------- Load Model --------
with open(MODEL_PATH, "rb") as f:
    obj = pickle.load(f)

model = obj["model"]
scaler = obj["scaler"]
features = obj["features"]

# -------- Prepare Input --------
def prepare_input(data: dict):
    df = pd.DataFrame([data])

    # Check missing columns
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Scale Time & Amount
    if "Time" in df.columns and "Amount" in df.columns:
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    return df[features]

# -------- Home --------
@app.route("/")
def home():
    return "<h2>Credit Card Fraud Detection API Running</h2>"

# -------- Predict API --------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = prepare_input(data)

        proba = model.predict_proba(X)[0][1]
        pred = int(model.predict(X)[0])

        return jsonify({
            "prediction": pred,
            "fraud_probability": float(proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------- Health Check --------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

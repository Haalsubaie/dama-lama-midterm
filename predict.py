
import pickle
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

# Load model
model_file = Path(__file__).parent / "model.bin"

with open(model_file, "rb") as f:
    dv, model, median_wait = pickle.load(f)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    patient = request.get_json(silent=True)

    if not isinstance(patient, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    # Check required fields
    required = [
        "Facility_Code",
        "Register_Date",
        "Age",
        "Triage_Level",
        "Previous_ED_Visits"
    ]

    for field in required:
        if patient.get(field) is None:
            return jsonify({"error": "Missing " + field}), 400

    # Fill missing values
    patient["Gender"] = patient.get("Gender") or "Unknown"
    patient["Arrival_Mode"] = patient.get("Arrival_Mode") or "Unknown"

    if patient.get("Waiting_Time_Minutes") is None:
        patient["Waiting_Time_Minutes"] = median_wait

    # Create date features
    try:
        date = pd.to_datetime(patient["Register_Date"], errors="raise")

        if pd.isna(date):
            raise ValueError("Invalid date")

        patient["Hour"] = date.hour
        patient["Day"] = date.dayofweek
        patient["Month"] = date.month

    except (ValueError, TypeError, OverflowError):
        return jsonify({"error": "Invalid Register_Date"}), 400

    # Select features
    features = [
        "Facility_Code",
        "Age",
        "Gender",
        "Triage_Level",
        "Arrival_Mode",
        "Waiting_Time_Minutes",
        "Previous_ED_Visits",
        "Hour",
        "Day",
        "Month"
    ]

    patient_data = {feature: patient[feature] for feature in features}

    # Make prediction
    X = dv.transform([patient_data])
    prediction = model.predict_proba(X)[0, 1]

    result = {
        "dama_lama_probability": float(prediction),
        "dama_lama": bool(prediction >= 0.5)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=9696)

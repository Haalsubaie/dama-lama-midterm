
import requests

# API URL
url = "http://127.0.0.1:9696/predict"

# Synthetic patient data
patient = {
    "Facility_Code": "Hospital_A",
    "Register_Date": "2025-07-27 14:00:00",
    "Age": 35,
    "Gender": "Female",
    "Triage_Level": 3,
    "Arrival_Mode": "Walk-in",
    "Waiting_Time_Minutes": 120,
    "Previous_ED_Visits": 2
}

# Test prediction API
try:
    response = requests.post(
        url,
        json=patient,
        timeout=15
    )

    print("Status Code:", response.status_code)
    print("Prediction:", response.json())

except requests.exceptions.Timeout:
    print("Error: API response took too long.")

except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to API. Check Docker.")

except ValueError:
    print("Error: Invalid API response.")

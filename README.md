# DynamicPricingSystem

Small Flask-based dynamic pricing demo that exposes a profit prediction pipeline using a Linear Regression model.  

This repository contains the web app and utilities to parse incoming payloads, engineer features, apply the 
preprocessor (if present), run the model, format output values (USD/INR), and persist prediction logs.

🌐 Check the project out here: https://dynamic-pricing-system-q546.onrender.com/

---

## Project layout

DynamicPricingSystem/

├── app.py   # Flask app entrypoint

├── requirements.txt # (project dependencies)

├── routes/

│     ├── init.py # blueprint registration

│     ├── main_routes.py # flask endpoints (prediction, UI)

│     └── utils.py # heavy lifting: parsing, preprocessing, model loader, helpers

├── models/

│     ├── LinearReg.pkl # Linear Regression model used for predictions

│     └── preprocessor.pkl # optional preprocessor (e.g. ColumnTransformer) — may or may not be present

├── data_entries/

│       └── prediction_log.csv # model request + response log (app appends here)

├── static/

├── templates/

└── tests/

      └── test_predictions.py # (not included by default) test script to exercise predictions


> **Note:** Many dev/IDE files and a `.venv` were included in the uploaded archive; they are not required to run the app. Focus on the files above.

---

## Key components

- `routes/utils.py`
  
  - `ModelLoader` — thread-safe lazy loader for `LinearReg.pkl` and optional `preprocessor.pkl`. Use the singleton `model_loader` instance to get artifacts.  
  
  - `parse_input` — validates and normalizes incoming payloads.  

  - `preprocess_input` — converts parsed inputs to the engineered features DataFrame expected by the preprocessor/model.  
  
  - `safe_preprocessor_transform` — calls `preprocessor.transform(df)` but will detect common "missing columns" errors and add zero-filled columns before retrying.  
  
  - `fit_to_n_features` / `to_matrix` / `align_features` — utilities to ensure the matrix shape matches what the model expects.  
  
  - `save_to_csv` — appends request + prediction rows to `data_entries/prediction_log.csv`.



- `routes/main_routes.py`  
  - Flask endpoints that accept JSON/form payloads, call `parse_input` → `preprocess_input` → (preprocessor) → model prediction → formatting → response. Also logs and persists predictions.

---

## Quick start (development)

1. Create & activate a virtual environment:

```bash
python -m venv .venv

# Unix/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
Install dependencies:


pip install -r requirements.txt
The requirements.txt included may be large or contain extras; for a minimal run you mostly need Flask, numpy, pandas, scikit-learn, and joblib. If you run into encoding or version issues, create a small virtualenv and install the exact libs your environment needs.

Confirm models exist:

models/LinearReg.pkl must exist.

models/preprocessor.pkl is optional. If missing, the code will try to run on engineered features (the utils include handling to align/pad features).

Run the app locally: python app.py

# App will start on: http://127.0.0.1:5000

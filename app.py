import os
import re
import numpy as np
import joblib
from flask import Flask, render_template, request
from PIL import Image
import pytesseract

app = Flask(__name__)

# Load core infrastructure safe tools
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def extract_metrics_from_ocr(image_file):
    """
    Scans the uploaded document text line-by-line using Regex
    to parse numbers next to diagnostic headers.
    """
    extracted_values = {}
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img).lower()
        
        # Regex engines looking for keys like: "chol: 240" or "blood pressure: 130"
        bp_match = re.search(r'(bp|pressure|trestbps)\D*(\d+)', text)
        chol_match = re.search(r'(chol|cholesterol)\D*(\d+)', text)
        hr_match = re.search(r'(heart rate|thalach|peak hr)\D*(\d+)', text)
        age_match = re.search(r'(age)\D*(\d+)', text)

        if bp_match: extracted_values['trestbps'] = float(bp_match.group(2))
        if chol_match: extracted_values['chol'] = float(chol_match.group(2))
        if hr_match: extracted_values['thalach'] = float(hr_match.group(2))
        if age_match: extracted_values['age'] = float(age_match.group(2))
    except Exception as e:
        print(f"OCR Scan failed safely: {e}")
    return extracted_values

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:

        uploaded_file = request.files.get("report_file")
        ocr_data = {}

        if uploaded_file and uploaded_file.filename:
            ocr_data = extract_metrics_from_ocr(uploaded_file)

        skipped = []

        def get_value(field, default, optional=False):

            manual = request.form.get(field, "").strip()

            if manual:
                return float(manual)

            if field in ocr_data:
                return ocr_data[field]

            if optional:
                skipped.append(field)

            return float(default)

        # Required fields
        age = get_value("age", 54)
        sex = get_value("sex", 1)
        cp = get_value("cp", 0)
        trestbps = get_value("trestbps", 131)
        chol = get_value("chol", 246)
        fbs = get_value("fbs", 0)
        exang = get_value("exang", 0)

        # Optional fields
        restecg = get_value("restecg", 0, True)
        thalach = get_value("thalach", 149, True)
        oldpeak = get_value("oldpeak", 0, True)
        slope = get_value("slope", 1, True)
        ca = get_value("ca", 0, True)
        thal = get_value("thal", 2, True)

        smoking = float(request.form.get("smoking", 0))
        family_history = float(request.form.get("family_history", 0))

        features = [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]

        data = np.array(features).reshape(1, -1)
        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data_scaled)[0][1]
            risk_pct = round(probability * 100)
        else:
            risk_pct = 40 if prediction == 0 else 75

        # Lifestyle adjustments
        if smoking:
            risk_pct += 8

        if family_history:
            risk_pct += 8

        risk_pct = min(risk_pct, 99)

        if risk_pct < 35:
            level = "low"

        elif risk_pct < 70:
            level = "medium"

        else:
            level = "high"

        if level == "low":
            prediction_text = "Your overall heart attack risk appears to be LOW."

        elif level == "medium":
            prediction_text = "Your overall heart attack risk appears to be MODERATE."

        else:
            prediction_text = "Your overall heart attack risk appears to be HIGH."

        return render_template(
            "result.html",
            prediction=prediction_text,
            level=level,
            risk_pct=risk_pct,
            skipped=skipped,
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction=str(e),
            level="error",
            risk_pct=None,
            skipped=[],
        )

if __name__ == '__main__':
    app.run(debug=True)
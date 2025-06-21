from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pandas as pd
import os

# Load model and preprocessors
rf = joblib.load("artefacts/rf_regressor_slim.pkl")
scaler = joblib.load("artefacts/scaler_slim.pkl")
label_encoders = joblib.load("artefacts/label_encoders_slim.pkl")

app = Flask(__name__)
CORS(app)

def grade_letter(score):
    if score >= 18: return "A+"
    if score >= 15: return "A"
    if score >= 12: return "B"
    if score >= 9: return "C"
    return "F"

@app.route("/", methods=["GET"])
def root():
    return "✅ Flask API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        student = request.json
        df = pd.DataFrame([student])
        for col, le in label_encoders.items():
            df[col] = le.transform(df[col].astype(str))
        X_scaled = scaler.transform(df.astype(float))
        g3 = rf.predict(X_scaled)[0]
        return jsonify({
        "predicted_g3": round(float(g3), 2),
        "grade": grade_letter(g3),
        "passed": bool(g3 >= 10)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

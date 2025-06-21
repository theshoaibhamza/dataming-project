
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask_cors import CORS

# Load artefacts
rf = joblib.load("artefacts/rf_regressor_slim.pkl")
scaler = joblib.load("artefacts/scaler_slim.pkl")
label_encoders = joblib.load("artefacts/label_encoders_slim.pkl")

app = Flask(__name__)
CORS(app)

def grade_letter(score):
    if score >= 16: return "A"
    if score >= 14: return "B"
    if score >= 12: return "C"
    if score >= 10: return "D (Pass)"
    return "F (Fail)"

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
            "passed": g3 >= 10
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def health():
    return "✅ Student Performance API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

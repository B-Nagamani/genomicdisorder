import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24)

BASE = Path(__file__).resolve().parent

# Load models and encoders globally
xgb_model = joblib.load(BASE / 'disorder_type_model_xgboost_10.joblib')
target_le = joblib.load(BASE / 'target_encoder_10.joblib')
subclass_model = joblib.load(BASE / 'subclass_model_xgboost_10.joblib')
subclass_le = joblib.load(BASE / 'subclass_encoder_10.joblib')
feature_encoders = joblib.load(BASE / 'feature_encoders_10.joblib')
feature_cols = joblib.load(BASE / 'top_10_features.joblib')

CLINICAL_FEATURES = {
    'Heart Rate (rates/min': 'numerical',
    'Respiratory Rate (breaths/min)': 'numerical',
    'White Blood cell count (thousand per microliter)': 'numerical',
    'Blood test result': ['abnormal', 'inconclusive', 'normal', 'slightly abnormal'],
    "Genes in mother's side": ['No', 'Yes'],
    'Inherited from father': ['No', 'Yes'],
    'Maternal gene': ['No', 'Yes'],
    'Paternal gene': ['No', 'Yes'],
    'History of anomalies in previous pregnancies': ['No', 'Yes'],
    'Assisted conception IVF/ART': ['No', 'Yes']
}

# In-memory mock database for users
users_db = {}

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in users_db and users_db[email] == password:
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in users_db:
            return render_template("register.html", error="User already exists.")
        users_db[email] = password
        session['user'] = email
        return redirect(url_for('dashboard'))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html", features=CLINICAL_FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    try:
        input_data = {}
        for col in feature_cols:
            input_data[col] = request.form.get(col, "")

        df = pd.DataFrame([input_data])

        for col in df.columns:
            if col in feature_encoders:
                le = feature_encoders[col]
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = -1 
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        pred_encoded = xgb_model.predict(df)[0]
        probas = xgb_model.predict_proba(df)[0]
        confidence = round(probas.max() * 100, 2)
        prediction_label = target_le.inverse_transform([pred_encoded])[0]

        sub_probas = subclass_model.predict_proba(df)[0]
        sub_classes = subclass_le.classes_
        subclass_dict = {sub_classes[i]: float(round(p * 100, 2)) for i, p in enumerate(sub_probas)}
        sorted_subclass = dict(sorted(subclass_dict.items(), key=lambda item: item[1], reverse=True)[:5])

        return render_template("result.html", prediction=prediction_label, confidence=confidence, subclasses=sorted_subclass)

    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
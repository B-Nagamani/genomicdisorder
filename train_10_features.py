import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TOP_10_FEATURES = [
    'Heart Rate (rates/min',
    'Respiratory Rate (breaths/min)',
    'White Blood cell count (thousand per microliter)',
    'Blood test result',
    "Genes in mother's side",
    'Inherited from father',
    'Maternal gene',
    'Paternal gene',
    'History of anomalies in previous pregnancies',
    'Assisted conception IVF/ART'
]

print("Loading data for Top 10 Features models...")
df = pd.read_csv('train.csv')
df = df.dropna(subset=['Genetic Disorder', 'Disorder Subclass'])

X = df[TOP_10_FEATURES].copy()
y_disorder = df['Genetic Disorder']
y_subclass = df['Disorder Subclass']

# Preprocess X
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna("Unknown").astype(str)
    else:
        X[col] = X[col].fillna(-1)

encoders = {}
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]) or X[col].dtype == 'object' or str(X[col].dtype) in ['string', 'category']:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col]).astype(int)
        encoders[col] = le
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)

joblib.dump(encoders, 'feature_encoders_10.joblib')
joblib.dump(TOP_10_FEATURES, 'top_10_features.joblib')

# Encode Targets
target_le = LabelEncoder()
y_disorder_encoded = target_le.fit_transform(y_disorder)
joblib.dump(target_le, 'target_encoder_10.joblib')

subclass_le = LabelEncoder()
y_subclass_encoded = subclass_le.fit_transform(y_subclass)
joblib.dump(subclass_le, 'subclass_encoder_10.joblib')

# Train Disorder Model
X_train, X_test, y_train, y_test = train_test_split(X, y_disorder_encoded, test_size=0.2, random_state=42)
model_disorder = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_disorder.fit(X_train, y_train)
joblib.dump(model_disorder, 'disorder_type_model_xgboost_10.joblib')
print("Disorder Model Acc:", model_disorder.score(X_test, y_test))

# Train Subclass Model
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_subclass_encoded, test_size=0.2, random_state=42)
model_subclass = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_subclass.fit(X_train_s, y_train_s)
joblib.dump(model_subclass, 'subclass_model_xgboost_10.joblib')
print("Subclass Model Acc:", model_subclass.score(X_test_s, y_test_s))

print("Saved new 10-feature models!")

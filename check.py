import joblib
encoders = joblib.load('feature_encoders.joblib')
for k in ["Genes in mother's side", "Inherited from father", "Maternal gene", "Paternal gene", "History of anomalies in previous pregnancies", "Assisted conception IVF/ART", "Blood test result"]:
    if k in encoders:
        print(f"{k}: {list(encoders[k].classes_)}")
    else:
        print(f"{k}: NOT CATEGORICAL")

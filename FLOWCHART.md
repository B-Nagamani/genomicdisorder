# Flowchart

Below is a simple diagram that shows how the pieces interact at runtime and during model training. Paste this into any Markdown viewer that supports Mermaid to see it rendered.

```mermaid
flowchart LR
  subgraph UI[User interface]
    A[Landing page] --> B[Login / Register]
    B --> C[Dashboard (top-10 features form)]
    C --> D[Submit prediction request]
  end

  subgraph S[Server (`app.py`)]
    D --> E[Validate & encode inputs]
    E --> F[XGBoost disorder model (`disorder_type_model_xgboost_10.joblib`)]
    E --> G[XGBoost subclass model (`subclass_model_xgboost_10.joblib`)]
    F --> H[Disorder label + confidence]
    G --> I[Subclass probabilities]
    H --> J[Render `result.html`]
    I --> J
  end

  subgraph T[Training]
    K[train.csv] --> L[`train_10_features.py`]
    L --> M[`feature_encoders_10.joblib` & `top_10_features.joblib`]
    L --> F
    L --> G
    N[`prepare_and_train.py`] --> O[full-feature artifacts & TabNet models]
  end
```

Notes:
- The diagram focuses on the top-10 feature path used by the running app. Full training scripts produce additional artifacts that can be integrated if needed.
- If you want this exported as an SVG/PNG I can render it and add the image to the repo.

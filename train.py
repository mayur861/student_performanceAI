"""
Student Performance Prediction - Training Pipeline
Target: math_score
Run: python train.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "stud.csv"
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "math_score"
X = df.drop(columns=[TARGET])

y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, random_state=42, max_depth=10, min_samples_leaf=2, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=400, random_state=42, max_depth=12, min_samples_leaf=2, n_jobs=-1
    ),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []
pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "model": name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    })
    pipelines[name] = pipe
    print(f"{name:20s} | MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

results_df = pd.DataFrame(results).sort_values("rmse")
best_name = results_df.iloc[0]["model"]
best_pipeline = pipelines[best_name]

joblib.dump(best_pipeline, ARTIFACTS / "student_performance_pipeline.joblib")
results_df.to_csv(ARTIFACTS / "model_results.csv", index=False)

metadata = {
    "target": TARGET,
    "best_model": best_name,
    "features": X.columns.tolist(),
    "categorical_features": categorical_cols,
    "numeric_features": numeric_cols,
    "dataset_rows": int(len(df)),
    "test_size": 0.20
}
with open(ARTIFACTS / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nBest model:", best_name)
print("\nModel comparison:")
print(results_df.to_string(index=False))
print(f"\nSaved: {ARTIFACTS / 'student_performance_pipeline.joblib'}")

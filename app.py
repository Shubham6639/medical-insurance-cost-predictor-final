from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
DATA = BASE / "insurance_raw.csv"
INDEX = BASE / "static" / "index.html"
FEATURES = ["Age", "BMI", "Smoker_Code", "Conditions"]

app = FastAPI(title="Medical Insurance Cost Predictor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class PredictionInput(BaseModel):
    age: float = Field(..., ge=18, le=64)
    bmi: float = Field(..., ge=10, le=60)
    smoker: bool
    conditions: int = Field(..., ge=0, le=3)

class Predictor:
    def __init__(self):
        self.df = self.clean(pd.read_csv(DATA))
        X = self.df[FEATURES]
        y = self.df["Premium"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        self.metrics = {}
        self.predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            self.predictions[name] = pred
            self.metrics[name] = {
                "R2": float(r2_score(y_test, pred)),
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
            }
        self.best_name = max(self.metrics, key=lambda n: self.metrics[n]["R2"])
        self.best_model = self.models[self.best_name]

    @staticmethod
    def clean(df):
        df = df.drop_duplicates().copy()
        df["Gender"] = df["Gender"].astype("string").str.strip().str.lower().map(
            {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
        )
        df["Smoker"] = df["Smoker"].astype("string").str.strip().str.lower().map(
            {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}
        )
        for c in ["Age", "BMI", "Conditions", "Premium"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[~df["Age"].between(18, 100), "Age"] = np.nan
        df.loc[~df["BMI"].between(10, 60), "BMI"] = np.nan
        df.loc[df["Premium"] <= 0, "Premium"] = np.nan
        df = df.dropna(subset=["Premium"])
        for c in ["Age", "BMI", "Conditions"]:
            df[c] = df[c].fillna(df[c].median())
        for c in ["Gender", "Smoker"]:
            df[c] = df[c].fillna(df[c].mode()[0])
        df["Conditions"] = df["Conditions"].astype(int)
        for c in ["Age", "BMI"]:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df["Smoker_Code"] = (df["Smoker"] == "Yes").astype(int)
        return df

    def predict(self, p: PredictionInput):
        row = pd.DataFrame([[p.age, p.bmi, int(p.smoker), p.conditions]], columns=FEATURES)
        value = float(self.best_model.predict(row)[0])
        return value

predictor = Predictor()

@app.get("/api/health")
def health():
    return {"status": "ok", "model": predictor.best_name}

@app.get("/api/meta")
def meta():
    importances = None
    if hasattr(predictor.best_model, "feature_importances_"):
        importances = dict(zip(FEATURES, [float(x) for x in predictor.best_model.feature_importances_]))
    return {
        "model": predictor.best_name,
        "rows_after_cleaning": int(len(predictor.df)),
        "features": FEATURES,
        "metrics": predictor.metrics,
        "feature_importance": importances,
        "currency_note": "The model predicts the synthetic dataset's premium in its original currency units; the UI can display USD or INR for presentation only.",
    }

@app.post("/api/predict")
def predict(p: PredictionInput):
    try:
        premium = predictor.predict(p)
        return {
            "premium": round(premium, 2),
            "model": predictor.best_name,
            "inputs": {"age": p.age, "bmi": p.bmi, "smoker": p.smoker, "conditions": p.conditions},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/what-if")
def what_if(age: float, bmi: float, smoker: bool, conditions: int, scenario: str):
    data = {"age": age, "bmi": bmi, "smoker": smoker, "conditions": conditions}
    if scenario == "non-smoker": data["smoker"] = False
    elif scenario == "smoker": data["smoker"] = True
    elif scenario == "no-conditions": data["conditions"] = 0
    elif scenario == "bmi-normal": data["bmi"] = 22.0
    else: raise HTTPException(status_code=400, detail="Unknown scenario")
    p = PredictionInput(**data)
    return {"scenario": scenario, "premium": round(predictor.predict(p), 2)}


@app.get("/api/curve")
def curve(kind: str, age: float = 35, bmi: float = 24.2, smoker: bool = False, conditions: int = 0):
    if kind == "age":
        xs = list(range(18, 65, 2))
        series = {}
        for sm in [False, True]:
            rows = pd.DataFrame([[x, bmi, int(sm), conditions] for x in xs], columns=FEATURES)
            series["Smoker" if sm else "Non-smoker"] = [round(float(v), 2) for v in predictor.best_model.predict(rows)]
        return {"x": xs, "series": series, "x_label": "Age (years)"}
    if kind == "bmi":
        xs = [round(x * 0.5, 1) for x in range(32, 93)]
        series = {}
        for sm in [False, True]:
            rows = pd.DataFrame([[age, x, int(sm), conditions] for x in xs], columns=FEATURES)
            series["Smoker" if sm else "Non-smoker"] = [round(float(v), 2) for v in predictor.best_model.predict(rows)]
        return {"x": xs, "series": series, "x_label": "BMI"}
    raise HTTPException(status_code=400, detail="kind must be age or bmi")

@app.get("/")
def index():
    return FileResponse(INDEX)

@app.get("/{path:path}")
def static_files(path: str):
    candidate = BASE / "static" / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(INDEX)

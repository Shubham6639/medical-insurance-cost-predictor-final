# Medical Insurance Cost Predictor

Full-stack PBL demo for the TY IT Data Science project.

## What is real here?
- The browser sends age, BMI, smoker status, and number of pre-existing conditions to a FastAPI backend.
- The backend cleans `insurance_raw.csv` using the same workflow documented in the project report.
- It trains Linear Regression, Random Forest, and Gradient Boosting using `random_state=42` and a 20% test split.
- The model with the highest R² is selected automatically. On the supplied dataset this is Gradient Boosting.
- Predictions shown in the UI come from that trained scikit-learn model, not from the old hard-coded JavaScript formula.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Deploy
The included `render.yaml`, `Dockerfile`, and `Procfile` make this ready for a Python web host such as Render. A public deployment still requires an authenticated hosting account/repository connection; no credentials are stored in this project.

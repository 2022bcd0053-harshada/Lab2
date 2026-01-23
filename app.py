from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Ensure the model path is correct relative to app.py
model_path = os.path.join("outputs", "model", "model.pkl")
model = joblib.load(model_path)

app = FastAPI(title="Wine Quality Prediction API")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: WineInput):
    features = np.array([[ 
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    prediction = int(model.predict(features)[0])

    return {
        "name": "Harshada Raut",
        "roll_no": "2022BCD0053",
        "wine_quality": prediction
    }

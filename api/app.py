from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app=FastAPI()
model=joblib.load("models/wine_model.pkl")
class WineFeatures(BaseModel):
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
    Id:int
@app.get("/")
def home():
    return {"message":"Wine quality prediction API"}

@app.post("/predict")
def predict(data:WineFeatures):
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
        data.alcohol,
        data.Id
    ]])

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
    
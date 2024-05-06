import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HeartAttack(BaseModel):
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalanch: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
MODEL_PATH = "/Users/anuragkotiyal/Desktop/Projects/Heart Attack Prediction/models/lr.pkl"
with open(f"{MODEL_PATH}", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"Health status:": "OK"}

@app.post("/predict")
def predict(data: HeartAttack):
    sample = pd.DataFrame((data.dict().values()), columns = data.dict().keys())    
    # test_sample = scaler.transform(sample)
    # prediction = model.predict(test_sample)
    
    return {"Prediction:", sample}
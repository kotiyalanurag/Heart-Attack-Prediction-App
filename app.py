import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() # initialising fastapi app [uvicorn filename:appname]

MODEL_PATH = "/app/models/lr.pkl" # path to model inside docker container

class HeartAttack(BaseModel): # data validation using pydantic BaseModel
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

with open('scaler.pkl', 'rb') as f: # load data scaler used during model training
    scaler = pickle.load(f)
    
with open(f"{MODEL_PATH}", "rb") as f: # load trained model - LR, KNN, SVC
    model = pickle.load(f)

@app.get("/") # health status check function
def read_root():
    
    return {"Health status:": "OK"}

@app.post("/predict") # make prediction for a sample function
def make_prediction(item:HeartAttack):
    
    sample = pd.DataFrame([item.model_dump()])  # save sample as a dataframe
    test_sample = scaler.transform(sample)   # scale sample using standard scaler
    prediction = model.predict(test_sample)    # predict using trained model                 
    
    class_name = {0: "low chance", 1: "high chance"}    # actual class labels for 0 & 1
    
    return {"Prediction": class_name[int(prediction)]}  # return predicted label as a dictionary
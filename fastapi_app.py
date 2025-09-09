"""
FastAPI application for Titanic survival prediction
Python translation of plumber.R
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import io
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Predictor API",
    description="API for predicting Titanic passenger survival",
    version="1.0.0"
)

# Load model at startup
model = None
feature_names = None

@app.on_event("startup")
async def load_model():
    global model, feature_names
    try:
        model = joblib.load('models/titanic_model.pkl')
        feature_names = joblib.load('models/titanic_model_features.pkl')
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Warning: Model not found. Please run model training first.")

# Pydantic models for request/response
class PassengerInput(BaseModel):
    pclass: str = Field(..., description="Passenger class: 1st, 2nd, or 3rd")
    sex: str = Field(..., description="Sex: male or female")
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sibsp: int = Field(..., description="Number of siblings/spouses", ge=0)
    parch: int = Field(..., description="Number of parents/children", ge=0)
    fare: float = Field(..., description="Fare paid", ge=0)
    embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    title: str = Field(..., description="Title: Mr, Mrs, Miss, Master, etc.")

class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float
    prediction_class: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocess input data for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply same encoding as training
    pclass_map = {'1st': 0, '2nd': 1, '3rd': 2}
    sex_map = {'male': 1, 'female': 0}
    embarked_map = {'S': 2, 'C': 0, 'Q': 1}
    title_map = {
        'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3,
        'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Major': 8,
        'Other': 9
    }
    
    df['pclass'] = df['pclass'].map(pclass_map)
    df['sex'] = df['sex'].map(sex_map)
    df['embarked'] = df['embarked'].map(embarked_map)
    df['title'] = df['title'].map(title_map)
    
    # Handle unknown values
    df = df.fillna(0)
    
    return df

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Titanic Survival Predictor API", "status": "running"}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(model)),
        "features": feature_names,
        "n_features": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(passenger: PassengerInput):
    """
    Predict survival for a single passenger
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        input_data = preprocess_input(passenger.dict())
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)[0]
        prediction_class = model.predict(input_data)[0]
        
        return PredictionResponse(
            survived=bool(prediction_class),
            survival_probability=float(prediction_proba[1]),
            prediction_class="Survived" if prediction_class == 1 else "Did not survive"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict survival for multiple passengers from CSV file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Process predictions
        predictions = []
        for _, row in df.iterrows():
            input_data = preprocess_input(row[required_cols].to_dict())
            prediction_proba = model.predict_proba(input_data)[0]
            prediction_class = model.predict(input_data)[0]
            
            predictions.append(PredictionResponse(
                survived=bool(prediction_class),
                survival_probability=float(prediction_proba[1]),
                prediction_class="Survived" if prediction_class == 1 else "Did not survive"
            ))
        
        # Calculate summary statistics
        total_passengers = len(predictions)
        survivors = sum(1 for p in predictions if p.survived)
        survival_rate = survivors / total_passengers if total_passengers > 0 else 0
        
        summary = {
            "total_passengers": total_passengers,
            "predicted_survivors": survivors,
            "survival_rate": survival_rate,
            "average_survival_probability": np.mean([p.survival_probability for p in predictions])
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/custom")
async def predict_custom(passengers: List[PassengerInput]):
    """
    Predict survival for multiple passengers from JSON input
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for passenger in passengers:
            input_data = preprocess_input(passenger.dict())
            prediction_proba = model.predict_proba(input_data)[0]
            prediction_class = model.predict(input_data)[0]
            
            predictions.append(PredictionResponse(
                survived=bool(prediction_class),
                survival_probability=float(prediction_proba[1]),
                prediction_class="Survived" if prediction_class == 1 else "Did not survive"
            ))
        
        # Calculate summary
        total_passengers = len(predictions)
        survivors = sum(1 for p in predictions if p.survived)
        survival_rate = survivors / total_passengers if total_passengers > 0 else 0
        
        summary = {
            "total_passengers": total_passengers,
            "predicted_survivors": survivors,
            "survival_rate": survival_rate,
            "average_survival_probability": np.mean([p.survival_probability for p in predictions])
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Custom prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Prediction endpoints for Heart Disease and Diabetes
"""

from fastapi import APIRouter, HTTPException
import joblib
import numpy as np
import os
import json

router = APIRouter()

# ============ LOAD MODELS AND FEATURES ============
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

# Load Heart Disease model and features
try:
    heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_disease_xgboost.pkl"))
    with open(os.path.join(MODEL_DIR, "heart_disease_features.json"), "r") as f:
        heart_features = json.load(f)
    print("✅ Heart disease model loaded")
except Exception as e:
    heart_model = None
    heart_features = None
    print(f"⚠️ Heart disease model not found: {e}")

# Load Diabetes model and features
try:
    diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_xgboost.pkl"))
    with open(os.path.join(MODEL_DIR, "diabetes_features.json"), "r") as f:
        diabetes_features = json.load(f)
    print("✅ Diabetes model loaded")
except Exception as e:
    diabetes_model = None
    diabetes_features = None
    print(f"⚠️ Diabetes model not found: {e}")


# ============ HELPER FUNCTION ============
def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


# ============ SCHEMAS (inline to avoid import issues) ============
from pydantic import BaseModel

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    trestbps: int
    chol: int
    fbs: int
    thalch: int
    exang: int
    oldpeak: float
    ca: int

class DiabetesInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    gender_encoded: int
    smoking_encoded: int

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: str


# ============ HEART DISEASE ENDPOINT ============
@router.post("/heart", response_model=PredictionResponse)
def predict_heart_disease(data: HeartDiseaseInput):
    """Predict heart disease risk"""
    
    if heart_model is None:
        raise HTTPException(status_code=503, detail="Heart disease model not loaded")
    
    # Create engineered features (matching training)
    hr_reserve = 220 - data.age - data.thalch
    risk_score = data.exang + data.ca + (1 if data.oldpeak > 1 else 0)
    chol_to_age = data.chol / data.age if data.age > 0 else 0
    bp_to_age = data.trestbps / data.age if data.age > 0 else 0
    chol_filled = data.chol
    
    # Build feature dict matching model's expected order
    feature_dict = {
        "exang": data.exang,
        "sex_clean": data.sex,
        "ca": data.ca,
        "chol": data.chol,
        "oldpeak": data.oldpeak,
        "thalch": data.thalch,
        "fbs": data.fbs,
        "hr_reserve": hr_reserve,
        "risk_score": risk_score,
        "chol_to_age": chol_to_age,
        "bp_to_age": bp_to_age,
        "chol_filled": chol_filled,
        "age": data.age,
        "trestbps": data.trestbps
    }
    
    # Create array in correct order
    features = np.array([[feature_dict[f] for f in heart_features]])
    
    # Make prediction
    prediction = heart_model.predict(features)[0]
    probability = heart_model.predict_proba(features)[0][1]
    
    return PredictionResponse(
        prediction="Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        probability=round(float(probability), 4),
        risk_level=get_risk_level(probability),
        confidence=f"{probability * 100:.1f}%"
    )


# ============ DIABETES ENDPOINT ============
@router.post("/diabetes", response_model=PredictionResponse)
def predict_diabetes(data: DiabetesInput):
    """Predict diabetes risk"""
    
    if diabetes_model is None:
        raise HTTPException(status_code=503, detail="Diabetes model not loaded")
    
    # Create engineered features (matching training)
    is_obese = 1 if data.bmi >= 30 else 0
    is_elderly = 1 if data.age >= 65 else 0
    high_glucose = 1 if data.blood_glucose_level > 140 else 0
    age_bmi = data.age * data.bmi
    glucose_bmi = data.blood_glucose_level * data.bmi
    risk_score = data.hypertension + data.heart_disease + is_obese
    hypertension_heart = data.hypertension * data.heart_disease
    age_squared = data.age ** 2
    bmi_squared = data.bmi ** 2
    
    # Build feature dict matching model's expected order
    feature_dict = {
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "bmi": data.bmi,
        "HbA1c_level": data.HbA1c_level,
        "blood_glucose_level": data.blood_glucose_level,
        "gender_encoded": data.gender_encoded,
        "smoking_encoded": data.smoking_encoded,
        "is_obese": is_obese,
        "is_elderly": is_elderly,
        "high_glucose": high_glucose,
        "age_bmi": age_bmi,
        "glucose_bmi": glucose_bmi,
        "risk_score": risk_score,
        "hypertension_heart": hypertension_heart,
        "age_squared": age_squared,
        "bmi_squared": bmi_squared
    }
    
    # Create array in correct order
    features = np.array([[feature_dict[f] for f in diabetes_features]])
    
    # Make prediction
    prediction = diabetes_model.predict(features)[0]
    probability = diabetes_model.predict_proba(features)[0][1]
    
    return PredictionResponse(
        prediction="Diabetes Detected" if prediction == 1 else "No Diabetes",
        probability=round(float(probability), 4),
        risk_level=get_risk_level(probability),
        confidence=f"{probability * 100:.1f}%"
    )

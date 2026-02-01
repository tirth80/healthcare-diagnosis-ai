"""
Request and Response schemas for the API
"""

from pydantic import BaseModel

# ============ HEART DISEASE ============
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int  # 0 = Female, 1 = Male
    trestbps: int  # Resting blood pressure
    chol: int  # Cholesterol
    fbs: int  # Fasting blood sugar (0 or 1)
    thalch: int  # Max heart rate achieved
    exang: int  # Exercise induced angina (0 or 1)
    oldpeak: float  # ST depression
    ca: int  # Number of major vessels (0-3)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "thalch": 150,
                "exang": 0,
                "oldpeak": 1.5,
                "ca": 0
            }
        }

# ============ DIABETES ============
class DiabetesInput(BaseModel):
    age: float
    hypertension: int  # 0 or 1
    heart_disease: int  # 0 or 1
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    gender_encoded: int  # 0 = Female, 1 = Male
    smoking_encoded: int  # 0-4

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45.0,
                "hypertension": 0,
                "heart_disease": 0,
                "bmi": 28.5,
                "HbA1c_level": 6.2,
                "blood_glucose_level": 140,
                "gender_encoded": 1,
                "smoking_encoded": 2
            }
        }

# ============ PREDICTION RESPONSE ============
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: str

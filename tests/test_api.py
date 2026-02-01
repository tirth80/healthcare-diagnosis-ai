"""
Tests for the Healthcare API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Create test client
client = TestClient(app)


# ============ HEALTH CHECK TESTS ============
def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============ HEART DISEASE TESTS ============
def test_predict_heart_low_risk():
    """Test heart disease prediction - low risk patient"""
    data = {
        "age": 35,
        "sex": 0,
        "trestbps": 120,
        "chol": 180,
        "fbs": 0,
        "thalch": 170,
        "exang": 0,
        "oldpeak": 0.0,
        "ca": 0
    }
    response = client.post("/predict/heart", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert "risk_level" in response.json()

def test_predict_heart_high_risk():
    """Test heart disease prediction - high risk patient"""
    data = {
        "age": 65,
        "sex": 1,
        "trestbps": 160,
        "chol": 300,
        "fbs": 1,
        "thalch": 120,
        "exang": 1,
        "oldpeak": 3.0,
        "ca": 2
    }
    response = client.post("/predict/heart", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["risk_level"] in ["Low", "Medium", "High"]


# ============ DIABETES TESTS ============
def test_predict_diabetes_low_risk():
    """Test diabetes prediction - low risk patient"""
    data = {
        "age": 30.0,
        "hypertension": 0,
        "heart_disease": 0,
        "bmi": 22.0,
        "HbA1c_level": 5.0,
        "blood_glucose_level": 100,
        "gender_encoded": 0,
        "smoking_encoded": 0
    }
    response = client.post("/predict/diabetes", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()

def test_predict_diabetes_high_risk():
    """Test diabetes prediction - high risk patient"""
    data = {
        "age": 65.0,
        "hypertension": 1,
        "heart_disease": 1,
        "bmi": 35.0,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 200,
        "gender_encoded": 1,
        "smoking_encoded": 2
    }
    response = client.post("/predict/diabetes", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["risk_level"] == "High"


# ============ VALIDATION TESTS ============
def test_invalid_heart_data():
    """Test heart disease with missing fields"""
    data = {"age": 55}  # Missing required fields
    response = client.post("/predict/heart", json=data)
    assert response.status_code == 422  # Validation error

def test_invalid_diabetes_data():
    """Test diabetes with missing fields"""
    data = {"age": 45.0}  # Missing required fields
    response = client.post("/predict/diabetes", json=data)
    assert response.status_code == 422  # Validation error

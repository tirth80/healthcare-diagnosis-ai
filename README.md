# 🏥 Healthcare Diagnosis AI System

An end-to-end AI system for medical diagnosis with **Heart Disease prediction**, **Diabetes prediction**, and **Pneumonia X-ray classification**. Deployed on **AWS EC2** with **FastAPI** and **Docker**.

## 🌐 Live API

**🔗 Try it now:** [http://3.145.57.206:8000/docs](http://3.145.57.206:8000/docs)

## 🎯 Project Overview

| Model | Data Type | Dataset Size | ROC-AUC | Key Achievement |
|-------|-----------|--------------|---------|-----------------|
| Heart Disease | Tabular | 920 patients | 0.87 | 85.33% accuracy |
| Diabetes | Tabular | 100,000 patients | 0.9796 | 94.1% recall |
| X-ray CNN | Images | 5,856 images | 0.9548 | 98.7% recall |

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    HEALTHCARE DIAGNOSIS AI                   │
├─────────────────────────────────────────────────────────────┤
│  Patient Data ──► XGBoost ──────────► Heart Disease Risk    │
│  Patient Data ──► XGBoost ──────────► Diabetes Risk         │
│  X-ray Images ──► CNN (ResNet18) ───► Pneumonia Detection   │
│  Explainability ► SHAP + Grad-CAM ──► Model Interpretability│
│  API Layer ─────► FastAPI ──────────► Docker + AWS EC2      │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Model Performance

### Heart Disease Prediction (XGBoost)
- **Accuracy:** 85.33%
- **ROC-AUC:** 0.87
- **Features:** 14 engineered features
- **Techniques:** Optuna tuning, feature selection, threshold tuning
- **Explainability:** SHAP values

### Diabetes Prediction (XGBoost)
- **ROC-AUC:** 0.9796
- **Recall:** 94.1% (catches 94% of diabetes cases)
- **Features:** 17 engineered features
- **Techniques:** Class balancing (scale_pos_weight), Optuna tuning
- **Explainability:** SHAP values

### Pneumonia X-ray Classification (CNN)
- **Test Accuracy:** 84.78%
- **ROC-AUC:** 0.9548
- **Pneumonia Recall:** 98.7% (missed only 5 out of 390 cases)
- **Architecture:** ResNet18 (Transfer Learning)
- **Techniques:** Data augmentation, class weighting
- **Explainability:** Grad-CAM heatmaps

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Classical ML** | XGBoost, Optuna, scikit-learn |
| **Deep Learning** | PyTorch, torchvision, ResNet18 |
| **Explainability** | SHAP, Grad-CAM |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Containerization** | Docker |
| **Cloud** | AWS EC2, Docker Hub |
| **Testing** | pytest |
| **Database** | DuckDB (SQL analytics) |

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/tirth80/healthcare-diagnosis-ai.git
cd healthcare-diagnosis-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.main:app --reload
```

### Docker
```bash
# Build and run
docker build -t healthcare-ai .
docker run -p 8000:8000 healthcare-ai
```

## 📁 Project Structure
```
healthcare-diagnosis-ai/
├── src/
│   └── api/              # FastAPI application
│       ├── main.py       # API entry point
│       ├── routes/       # Prediction endpoints
│       └── schemas.py    # Request/Response models
├── models/               # Trained model files (.pkl, .json)
├── notebooks/            # Jupyter notebooks (EDA, training)
├── tests/                # pytest test files
├── data/                 # SQL queries
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict/heart` | POST | Heart disease prediction |
| `/predict/diabetes` | POST | Diabetes prediction |
| `/docs` | GET | Interactive API documentation |

## 📈 Skills Demonstrated

- **Machine Learning:** XGBoost, hyperparameter tuning, feature engineering
- **Deep Learning:** CNN, transfer learning, PyTorch
- **MLOps:** Docker, AWS EC2, API development
- **Data Engineering:** SQL (DuckDB), data pipelines
- **Software Engineering:** Testing (pytest), clean code structure
- **Explainable AI:** SHAP, Grad-CAM

## 👤 Author

**Tirth Patel**
- GitHub: [@tirth80](https://github.com/tirth80)

## 📄 License

This project is licensed under the MIT License.

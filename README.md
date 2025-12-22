# 🏥 Healthcare Diagnosis AI System

An AI system that analyzes medical images (X-rays) and patient health data to predict diseases and patient risk scores.

## 🎯 Project Overview

This project combines:
- **Deep Learning (PyTorch)**: CNN for chest X-ray classification
- **Classical ML (XGBoost)**: Patient risk prediction from tabular data
- **Explainability**: SHAP values + Grad-CAM visualizations
- **Production API**: FastAPI with Docker deployment

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    HEALTHCARE DIAGNOSIS AI                   │
├─────────────────────────────────────────────────────────────┤
│  X-ray Images ──► CNN (ResNet50) ──► Disease Prediction     │
│  Patient Data ──► XGBoost ──────────► Risk Score            │
│  Both Models ───► Explainability ───► SHAP + Grad-CAM       │
│  API Layer ─────► FastAPI ──────────► Docker + AWS          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure
```
healthcare-diagnosis-ai/
├── data/           # Raw and processed datasets
├── notebooks/      # Jupyter notebooks for exploration
├── src/            # Production source code
├── tests/          # Unit and integration tests
├── models/         # Saved model artifacts
├── configs/        # YAML configuration files
├── infrastructure/ # Docker and AWS configs
└── docs/           # Documentation
```

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-diagnosis-ai.git
cd healthcare-diagnosis-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| ChestX-ray14 | NIH (Kaggle) | X-ray classification |
| Heart Disease | UCI ML Repository | Risk prediction |
| Diabetes | Kaggle | Risk scoring |

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision, timm
- **Classical ML**: XGBoost, LightGBM, scikit-learn
- **Explainability**: SHAP, Grad-CAM
- **API**: FastAPI, Pydantic, Uvicorn
- **Database**: DuckDB, SQLAlchemy
- **Deployment**: Docker, AWS (EC2, S3)
- **CI/CD**: GitHub Actions

## 📈 Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| CNN (ResNet50) | X-ray Classification | Accuracy | TBD |
| XGBoost | Risk Prediction | ROC-AUC | TBD |

## 👤 Author

**Tirth Patel**

## 📄 License

This project is licensed under the MIT License.

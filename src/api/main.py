"""
Healthcare Diagnosis AI - FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from src.api.routes import predict

# Create FastAPI app
app = FastAPI(
    title="Healthcare Diagnosis AI",
    description="API for Heart Disease, Diabetes, and X-ray Pneumonia prediction",
    version="1.0.0"
)

# Allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])

# Health check endpoint
@app.get("/")
def root():
    return {"message": "Healthcare Diagnosis AI API is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "heart_disease": "available",
            "diabetes": "available",
            "xray_cnn": "coming soon"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

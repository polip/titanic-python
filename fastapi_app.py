import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage
import joblib
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
feature_names = None

def download_from_gcs(bucket_name, source_path, destination_path):
    """Download file from Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_path)
        
        # Check if blob exists
        if not blob.exists():
            logger.error(f"File not found: gs://{bucket_name}/{source_path}")
            return False
            
        blob.download_to_filename(destination_path)
        logger.info(f"Downloaded gs://{bucket_name}/{source_path} to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return False

def load_model_from_storage(bucket_name, model_path, features_path):
    """Load model and features from Google Cloud Storage"""
    try:
        logger.info(f"Loading model from Cloud Storage bucket: {bucket_name}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as model_file, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as features_file:
            
            model_loaded = False
            features_loaded = False
            loaded_model = None
            loaded_features = None
            
            # Download and load model
            if download_from_gcs(bucket_name, model_path, model_file.name):
                try:
                    loaded_model = joblib.load(model_file.name)
                    model_loaded = True
                    logger.info("✅ Model loaded from Cloud Storage")
                except Exception as e:
                    logger.error(f"Error loading model file: {e}")
            
            # Download and load features
            if download_from_gcs(bucket_name, features_path, features_file.name):
                try:
                    loaded_features = joblib.load(features_file.name)
                    features_loaded = True
                    logger.info("✅ Features loaded from Cloud Storage")
                except Exception as e:
                    logger.error(f"Error loading features file: {e}")
            
            # Clean up temp files
            try:
                os.unlink(model_file.name)
                os.unlink(features_file.name)
            except Exception as e:
                logger.warning(f"Could not clean up temp files: {e}")
            
            if model_loaded and features_loaded:
                return loaded_model, loaded_features
            else:
                logger.warning("Failed to load model or features from Cloud Storage")
                return None, None
                
    except Exception as e:
        logger.error(f"Error loading from Cloud Storage: {e}")
        return None, None

def create_dummy_model():
    """Create a dummy model for testing when real model is not available"""
    logger.info("Creating dummy model for testing")
    from sklearn.dummy import DummyClassifier
    import numpy as np
    
    dummy_model = DummyClassifier(strategy="constant", constant=1)
    dummy_model.fit(np.array([[1, 0, 25, 0, 0, 50, 0, 0]]), np.array([1]))
    dummy_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title']
    
    return dummy_model, dummy_features

async def load_model():
    """Load model from Google Cloud Storage or local files"""
    global model, feature_names
    
    try:
        bucket_name = os.getenv('MODEL_BUCKET')
        model_path = os.getenv('MODEL_PATH', 'titanic_model.pkl')
        features_path = os.getenv('FEATURES_PATH', 'titanic_model_features.pkl')
        
        # Try to load from Cloud Storage first
        if bucket_name and bucket_name != 'not_configured':
            logger.info(f"Attempting to load model from Cloud Storage bucket: {bucket_name}")
            model, feature_names = load_model_from_storage(bucket_name, model_path, features_path)
            
            if model is not None and feature_names is not None:
                logger.info("✅ Successfully loaded model from Cloud Storage")
                return
        
        # Try to load from local files as fallback
        local_model_path = os.path.join(os.getcwd(), 'models', 'titanic_model.pkl')
        local_features_path = os.path.join(os.getcwd(), 'models', 'titanic_model_features.pkl')
        
        if os.path.exists(local_model_path) and os.path.exists(local_features_path):
            logger.info("Loading model from local files")
            try:
                model = joblib.load(local_model_path)
                feature_names = joblib.load(local_features_path)
                logger.info("✅ Successfully loaded model from local files")
                return
            except Exception as e:
                logger.error(f"Error loading local model files: {e}")
        
        # Use dummy model as last resort
        logger.warning("Using dummy model for testing (no real model found)")
        model, feature_names = create_dummy_model()
            
    except Exception as e:
        logger.error(f"Error in model loading: {e}")
        # Use dummy model as fallback
        logger.warning("Using dummy model due to error")
        model, feature_names = create_dummy_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan manager - handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting up Titanic FastAPI application...")
    await load_model()
    logger.info("✅ Application startup complete")
    
    yield  # This is where the application runs
    
    # Shutdown (optional cleanup)
    logger.info("🛑 Shutting down Titanic FastAPI application...")
    # Add any cleanup code here if needed
    logger.info("✅ Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Titanic Survival Predictor API",
    description="API for predicting Titanic passenger survival",
    version="1.0.0",
    lifespan=lifespan  # ✅ Modern way to handle startup/shutdown
)

# Pydantic models
class PassengerInput(BaseModel):
    pclass: str = Field(..., description="Passenger class: 1st, 2nd, or 3rd")
    sex: str = Field(..., description="Sex: male or female")
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sibsp: int = Field(0, description="Number of siblings/spouses", ge=0)
    parch: int = Field(0, description="Number of parents/children", ge=0)
    fare: float = Field(..., description="Fare paid", ge=0)
    embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    title: str = Field(..., description="Title: Mr, Mrs, Miss, Master, etc.")

class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float
    prediction_class: str
    model_source: str

@app.get("/")
async def root():
    return {
        "message": "Titanic Survival Predictor API", 
        "status": "running",
        "deployment": "Google Cloud Run",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "cloud-run",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0,
        "bucket": os.getenv('MODEL_BUCKET', 'not_configured')
    }

@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables"""
    return {
        "MODEL_BUCKET": os.getenv('MODEL_BUCKET', 'NOT_SET'),
        "MODEL_PATH": os.getenv('MODEL_PATH', 'NOT_SET'),
        "FEATURES_PATH": os.getenv('FEATURES_PATH', 'NOT_SET'),
        "PORT": os.getenv('PORT', 'NOT_SET'),
        "GOOGLE_CLOUD_PROJECT": os.getenv('GOOGLE_CLOUD_PROJECT', 'NOT_SET')
    }

@app.post("/reload-model")
async def reload_model():
    """Manually reload the model from Cloud Storage"""
    global model, feature_names
    
    try:
        bucket_name = os.getenv('MODEL_BUCKET')
        
        if not bucket_name or bucket_name == 'NOT_SET' or bucket_name == 'not_configured':
            return {
                "success": False,
                "message": "MODEL_BUCKET environment variable not configured",
                "current_model": "dummy"
            }
        
        model_path = os.getenv('MODEL_PATH', 'models/titanic_model.pkl')
        features_path = os.getenv('FEATURES_PATH', 'models/titanic_model_features.pkl')
        
        # Attempt to load from storage
        new_model, new_features = load_model_from_storage(bucket_name, model_path, features_path)
        
        if new_model is not None and new_features is not None:
            model = new_model
            feature_names = new_features
            return {
                "success": True,
                "message": "Model successfully reloaded from Cloud Storage",
                "bucket": bucket_name,
                "model_path": model_path,
                "features_path": features_path,
                "features_count": len(feature_names)
            }
        else:
            return {
                "success": False,
                "message": "Failed to load model from Cloud Storage",
                "bucket": bucket_name,
                "current_model": "dummy" if hasattr(model, '_estimator_type') else "unknown"
            }
            
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return {
            "success": False,
            "message": f"Error reloading model: {str(e)}",
            "current_model": "dummy" if hasattr(model, '_estimator_type') else "unknown"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(passenger: PassengerInput):
    """Predict passenger survival"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Simple encoding (adjust based on your model)
        pclass_map = {'1st': 0, '2nd': 1, '3rd': 2}
        sex_map = {'male': 1, 'female': 0}
        embarked_map = {'S': 2, 'C': 0, 'Q': 1}
        title_map = {
            'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3,
            'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Major': 8,
            'Other': 9
        }
        
        # Create feature vector
        features = [
            pclass_map.get(passenger.pclass, 2),
            sex_map.get(passenger.sex, 0),
            passenger.age,
            passenger.sibsp,
            passenger.parch,
            passenger.fare,
            embarked_map.get(passenger.embarked, 2),
            title_map.get(passenger.title, 9)
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([features])[0]
            survival_prob = float(probability[1]) if len(probability) > 1 else 0.5
        else:
            survival_prob = 0.7 if prediction else 0.3
        
        model_source = "Cloud Storage" if os.getenv('MODEL_BUCKET') else "Dummy Model"
        
        return PredictionResponse(
            survived=bool(prediction),
            survival_probability=round(survival_prob, 3),
            prediction_class="Survived" if prediction else "Did not survive",
            model_source=model_source
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
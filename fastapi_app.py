import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage
import joblib
import tempfile
from typing import Tuple, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and tracking
model = None
feature_names = None
model_source_flag = None  # Track: 'gcs', 'local', 'dummy'

load_dotenv()  

# Load environment variables from .env file
# Environment variables needed
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
REGION = os.environ.get('GOOGLE_CLOUD_REGION')
BUCKET_NAME = os.environ.get('BUCKET_NAME')  
MODEL_FILE = os.environ.get('MODEL_FILE')     
FEATURES_FILE = os.environ.get('FEATURES_FILE')  
LOCAL_MODEL_DIR = os.environ.get('LOCAL_MODEL_DIR')

def download_and_load_from_gcs(bucket_name: str, source_path: str) -> Optional[object]:
    """
    Download and load file from Google Cloud Storage    
    Args: bucket_name: GCS bucket name
        source_path: Path to file in bucket
     Returns: Loaded object or None if failed
    """
    try:
        logger.info(f"📥 Downloading gs://{bucket_name}/{source_path}")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_path)
        
        # Check if blob exists
        if not blob.exists():
            logger.error(f"❌ File not found: gs://{bucket_name}/{source_path}")
            return None
            
        # Download to temporary file and load immediately
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            try:
                blob.download_to_filename(temp_file.name)
                logger.info(f"✅ Downloaded gs://{bucket_name}/{source_path}")
                
                # Load the file
                loaded_object = joblib.load(temp_file.name)
                logger.info(f"✅ Successfully loaded from gs://{bucket_name}/{source_path}")
                return loaded_object
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up temp file: {cleanup_error}")
                    
    except Exception as e:
        logger.error(f"❌ Error downloading/loading from GCS: {e}")
        return None

def load_from_gcs() -> Tuple[Optional[object], Optional[list]]:
    """
    Load model and features from Google Cloud Storage 
        Returns:
        Tuple of (model, features) or (None, None) if failed
    """
    try:
        logger.info(f"🌩️ Loading from Cloud Storage bucket: {BUCKET_NAME}")
        
        # Load model using hardcoded path
        loaded_model = download_and_load_from_gcs(BUCKET_NAME, MODEL_FILE)
        if loaded_model is None:
            logger.error("❌ Failed to load model from GCS")
            return None, None
            
        # Load features using hardcoded path
        loaded_features = download_and_load_from_gcs(BUCKET_NAME, FEATURES_FILE)
        if loaded_features is None:
            logger.error("❌ Failed to load features from GCS")
            return None, None
        
        logger.info("✅ Both model and features loaded from Cloud Storage")
        return loaded_model, loaded_features
                
    except Exception as e:
        logger.error(f"❌ Error loading from Cloud Storage: {e}")
        return None, None

def load_from_local() -> Tuple[Optional[object], Optional[list]]:
    """
    Load model and features from local files using hardcoded paths
    
    Returns:
        Tuple of (model, features) or (None, None) if failed
    """
    try:
        # local file paths
        local_model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_FILE)
        local_features_path = os.path.join(LOCAL_MODEL_DIR, FEATURES_FILE)
        
        if not (os.path.exists(local_model_path) and os.path.exists(local_features_path)):
            logger.info(f"📁 Local model files not found in: {LOCAL_MODEL_DIR}")
            return None, None
        
        logger.info(f"💾 Loading model from local files: {LOCAL_MODEL_DIR}")
        loaded_model = joblib.load(local_model_path)
        loaded_features = joblib.load(local_features_path)
        
        logger.info("✅ Successfully loaded model from local files")
        return loaded_model, loaded_features
        
    except Exception as e:
        logger.error(f"❌ Error loading local model files: {e}")
        return None, None

def create_dummy_model() -> Tuple[object, list]:
    """
    Create a dummy model for testing when real model is not available
    
    Returns:
        Tuple of (dummy_model, dummy_features)
    """
    logger.warning("🤖 Creating dummy model for testing")
    from sklearn.dummy import DummyClassifier
    import numpy as np
    
    dummy_model = DummyClassifier(strategy="constant", constant=1)
    dummy_model.fit(np.array([[1, 0, 25, 0, 0, 50, 0, 0]]), np.array([1]))
    dummy_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title']
    
    logger.info("✅ Dummy model created")
    return dummy_model, dummy_features

def get_model_source() -> str:
    """Get the source of the current model"""
    global model_source_flag
    
    if not model:
        return "No Model Loaded"
    
    if model_source_flag == 'gcs':
        return "Google Cloud Storage"
    elif model_source_flag == 'local':
        return "Local Files"
    elif model_source_flag == 'dummy':
        return "Dummy Model (Testing)"
    
    return "Unknown Source"

async def load_model():
    """
    Load model with priority order:
    1. Google Cloud Storage ( bucket and paths)
    2. Local files ( directory)
    3. Dummy model (last resort)
    """
    global model, feature_names, model_source_flag
    
    logger.info("🚀 Starting model loading process...")
    logger.info(f"📋 Target GCS Bucket: {BUCKET_NAME}")
    logger.info(f"📋 Target Model Path: {MODEL_FILE}")
    logger.info(f"📋 Target Features Path: {FEATURES_FILE}")
    logger.info(f"📋 Local Directory: {LOCAL_MODEL_DIR}")
    
    try:
        # Priority 1: Try Google Cloud Storage with hardcoded paths
        logger.info("🌩️ Attempting to load from Google Cloud Storage...")
        model, feature_names = load_from_gcs()
        
        if model is not None and feature_names is not None:
            model_source_flag = 'gcs'
            logger.info("✅ Model loaded from Google Cloud Storage")
            return
        
        # Priority 2: Try local files with hardcoded paths
        logger.info("💾 Attempting to load from local files...")
        model, feature_names = load_from_local()
        
        if model is not None and feature_names is not None:
            model_source_flag = 'local'
            logger.info("✅ Model loaded from local files")
            return
        
        # Priority 3: Create dummy model
        logger.warning("⚠️ No real model found, creating dummy model...")
        model, feature_names = create_dummy_model()
        model_source_flag = 'dummy'
            
    except Exception as e:
        logger.error(f"❌ Critical error in model loading: {e}")
        # Emergency fallback
        logger.warning("🚨 Using dummy model due to critical error")
        model, feature_names = create_dummy_model()
        model_source_flag = 'dummy'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager """
    # Startup
    logger.info("🚀 Starting Titanic FastAPI application...")
    logger.info(f"   - Project: {PROJECT_ID}")
    logger.info(f"   - Region: {REGION}")
    logger.info(f"   - Bucket: {BUCKET_NAME}")
    logger.info(f"   - Model Path: {MODEL_FILE}")
    logger.info(f"   - Features Path: {FEATURES_FILE}")
    
    await load_model()
    logger.info(f"✅ Application startup complete - Model Source: {get_model_source()}")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="🚢 Titanic Survival Predictor API",
    description="""
    ## Titanic Survival Prediction API 
    
    This API uses Google Cloud Storage bucket and file paths:
          
    ### Model Loading Priority:
    1. **Google Cloud Storage** 
    2. **Local Files** (directory)
    3. **Dummy Model** (testing fallback)
        
    """,
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models
class PassengerInput(BaseModel):
    """Input model for passenger data"""
    pclass: str = Field(..., description="Passenger class: 1st, 2nd, or 3rd")
    sex: str = Field(..., description="Sex: male or female")
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sibsp: int = Field(0, description="Number of siblings/spouses aboard", ge=0)
    parch: int = Field(0, description="Number of parents/children aboard", ge=0)
    fare: float = Field(..., description="Fare paid", ge=0)
    embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    title: str = Field(..., description="Title: Mr, Mrs, Miss, Master, etc.")

    class Config:
        json_schema_extra = {
            "example": {
                "pclass": "1st",
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 211.34,
                "embarked": "S",
                "title": "Miss"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    survived: bool
    survival_probability: float
    prediction_class: str
    model_source: str

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with hardcoded configuration info"""
    return {
        "message": "🚢 Titanic Survival Predictor API (Hardcoded Config)",
        "status": "running",
        "deployment": "Google Cloud Run",
        "model_loaded": model is not None,
        "model_source": get_model_source(),
        "configuration": "Hardcoded (No Environment Variables)",
        "version": "2.1.0-hardcoded",
        "hardcoded_config": {
            "bucket": BUCKET_NAME,
            "model_path": MODEL_FILE,
            "features_path": FEATURES_FILE,
            "local_dir": LOCAL_MODEL_DIR
        }
    }

@app.get("/health")
async def health_check():
    """Health check with hardcoded configuration details"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "service": "titanic-fastapi-hardcoded",
        "deployment": "google-cloud-run",
        "model_loaded": model is not None,
        "model_source": get_model_source(),
        "features_count": len(feature_names) if feature_names else 0,
        "configuration": "hardcoded",
        "hardcoded_paths": {
            "bucket": BUCKET_NAME,
            "model": MODEL_FILE,
            "features": FEATURES_FILE,
            "local": LOCAL_MODEL_DIR
        }
    }

@app.get("/config")
async def get_configuration():
    """Get configuration details"""
    local_model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_FILE )
    local_features_path = os.path.join(LOCAL_MODEL_DIR, FEATURES_FILE)
    
    return {
        "configuration_type": "HARDCODED",
        "google_cloud": {
            "project_id": PROJECT_ID,
            "region": REGION,
            "bucket_name": BUCKET_NAME,
            "model_path": MODEL_FILE,
            "features_path": FEATURES_FILE
        },
        "local_fallback": {
            "directory": LOCAL_MODEL_DIR,
            "model_path": local_model_path,
            "features_path": local_features_path,
            "local_files_exist": (os.path.exists(local_model_path) and 
                                 os.path.exists(local_features_path))
        },
        "current_model": {
            "source": get_model_source(),
            "type": type(model).__name__ if model else None,
            "features_count": len(feature_names) if feature_names else 0
        }
    }

@app.post("/reload-model")
async def reload_model():
    """Manually reload model using hardcoded configuration"""
    try:
        logger.info("🔄 Manual model reload requested (using hardcoded config)")
        old_source = get_model_source()
        
        await load_model()
        
        return {
            "success": True,
            "message": "Model reloaded successfully using hardcoded configuration",
            "old_source": old_source,
            "new_source": get_model_source(),
            "features_count": len(feature_names) if feature_names else 0,
            "hardcoded_config": {
                "bucket": BUCKET_NAME,
                "model_path": MODEL_FILE,
                "features_path": FEATURES_FILE,
            }
        }
            
    except Exception as e:
        logger.error(f"❌ Error reloading model: {e}")
        return {
            "success": False,
            "message": f"Error reloading model: {str(e)}",
            "current_source": get_model_source()
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(passenger: PassengerInput):
    """
    Predict passenger survival using hardcoded model configuration
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Check /health endpoint for details."
        )
    
    try:
        # Feature encoding mappings
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
        
        logger.info(f"🔮 Making prediction with features: {features}")
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([features])[0]
            survival_prob = float(probability[1]) if len(probability) > 1 else 0.5
        else:
            # Fallback for models without probability prediction
            survival_prob = 0.7 if prediction else 0.3
        
        logger.info(f"✅ Prediction: {prediction}, Probability: {survival_prob:.3f}")
        
        return PredictionResponse(
            survived=bool(prediction),
            survival_probability=round(survival_prob, 3),
            prediction_class="Survived" if prediction else "Did not survive",
            model_source=get_model_source()
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
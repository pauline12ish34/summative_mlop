"""
FastAPI Application for Shoe Classification
Provides endpoints for prediction, retraining, and monitoring
"""

import os
import json
import shutil
import threading
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import custom modules
import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import DataPreprocessor
from model import ShoeClassifier
from prediction import ShoePredictor


# Initialize FastAPI app
app = FastAPI(
    title="Shoe Classification API",
    description="ML Pipeline for Shoe Classification with Retraining Capability",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'shoe_classifier_model.h5')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

# Global state
predictor = None
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Idle',
    'last_trained': None
}
model_metrics = {}
request_count = 0
start_time = datetime.now()


# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    timestamp: str


class StatusResponse(BaseModel):
    status: str
    uptime: str
    total_requests: int
    model_loaded: bool
    training_status: dict
    model_metrics: Optional[dict]


class RetrainResponse(BaseModel):
    message: str
    status: str


# Helper functions
def load_predictor():
    """Load the predictor model"""
    global predictor, model_metrics
    if os.path.exists(MODEL_PATH):
        predictor = ShoePredictor(MODEL_PATH, CLASS_NAMES_PATH)
        # Load metrics if available
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, 'r') as f:
                    model_metrics.update(json.load(f))
            except:
                pass
        return True
    return False


def retrain_model_task(upload_path: str):
    """Background task for model retraining"""
    global training_status, predictor, model_metrics
    
    # Import TensorFlow here to ensure proper thread context
    import tensorflow as tf
    
    try:
        training_status['is_training'] = True
        training_status['progress'] = 10
        training_status['message'] = 'Preprocessing data...'
        
        # Clear any existing TensorFlow session
        tf.keras.backend.clear_session()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Check directory structure
        classes_found = [d for d in os.listdir(upload_path) 
                        if os.path.isdir(os.path.join(upload_path, d))]
        expected_classes = ['Boot', 'Sandal', 'Shoe']
        
        # Creating missing class directories with at least one image from existing data
        dataset_path = os.path.join(BASE_DIR, 'dataset', 'split_data', 'train')
        for cls in expected_classes:
            cls_path = os.path.join(upload_path, cls)
            if cls not in classes_found and os.path.exists(dataset_path):
                # Creating directory and copy one sample from original dataset
                os.makedirs(cls_path, exist_ok=True)
                source_cls_path = os.path.join(dataset_path, cls)
                if os.path.exists(source_cls_path):
                    # Copy first image found
                    for img_file in os.listdir(source_cls_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            source_img = os.path.join(source_cls_path, img_file)
                            dest_img = os.path.join(cls_path, img_file)
                            shutil.copy(source_img, dest_img)
                            break
        
        # Create data generators with explicit classes
        train_gen, val_gen = preprocessor.create_data_generators(
            upload_path,
            validation_split=0.2
        )
        
        training_status['progress'] = 30
        training_status['message'] = 'Loading existing model...'
        
        # Initialize model
        classifier = ShoeClassifier()
        
        if os.path.exists(MODEL_PATH):
            # Retrain existing model
            training_status['message'] = 'Retraining model...'
            history = classifier.retrain(
                train_gen, val_gen,
                MODEL_PATH,
                epochs=5  # Reduced for faster retraining
            )
        else:
            # Train new model
            training_status['message'] = 'Training new model...'
            classifier.build_model(use_pretrained=False)
            history = classifier.train(
                train_gen, val_gen,
                epochs=10,
                patience=3
            )
        
        training_status['progress'] = 80
        training_status['message'] = 'Evaluating model...'
        
        # Evaluate model
        metrics = classifier.evaluate(val_gen)
        model_metrics.update(metrics)
        
        # Save metrics to file
        with open(METRICS_PATH, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        training_status['progress'] = 90
        training_status['message'] = 'Saving model...'
        
        # Save model
        classifier.save_model(MODEL_DIR)
        
        training_status['progress'] = 95
        training_status['message'] = 'Reloading predictor...'
        
        # Clear session before reloading
        tf.keras.backend.clear_session()
        
        # Reload predictor
        load_predictor()
        
        training_status['progress'] = 100
        training_status['message'] = 'Training completed successfully!'
        training_status['last_trained'] = datetime.now().isoformat()
        
    except Exception as e:
        import traceback
        error_msg = f'Training failed: {str(e)}'
        print(f"{error_msg}\n{traceback.format_exc()}")
        training_status['message'] = error_msg
        training_status['progress'] = 0
    finally:
        training_status['is_training'] = False


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global start_time
    start_time = datetime.now()
    if load_predictor():
        print("Model loaded successfully!")
    else:
        print("Warning: No model found. Train a model first.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Shoe Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "upload_data": "/upload-data",
            "retrain": "/retrain",
            "status": "/status",
            "metrics": "/metrics"
        }
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API and model status"""
    global request_count, start_time
    
    uptime = datetime.now() - start_time
    uptime_str = str(uptime).split('.')[0]
    
    return StatusResponse(
        status="online",
        uptime=uptime_str,
        total_requests=request_count,
        model_loaded=predictor is not None,
        training_status=training_status,
        model_metrics=model_metrics if model_metrics else None
    )


@app.get("/metrics")
async def get_metrics():
    """Get model evaluation metrics"""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="No metrics available")
    return model_metrics


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict shoe class from uploaded image
    """
    global request_count, predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = predictor.predict_from_bytes(image_bytes)
        
        # Increment request count
        request_count += 1
        
        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/upload-data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    category: str = "Shoe"
):
    """
    Upload images for retraining
    Images should be organized by category
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create category directory
    category_dir = os.path.join(UPLOAD_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    saved_files = []
    
    for file in files:
        # Validate file type
        if not file.content_type.startswith('image/'):
            continue
        
        # Save file
        file_path = os.path.join(category_dir, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        saved_files.append(file.filename)
    
    return {
        "message": f"Uploaded {len(saved_files)} images to category '{category}'",
        "saved_files": saved_files,
        "category": category
    }


@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with uploaded data
    """
    if training_status['is_training']:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Check if upload directory has data
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No training data uploaded")
    
    # Start retraining in background
    background_tasks.add_task(retrain_model_task, UPLOAD_DIR)
    
    return RetrainResponse(
        message="Model retraining started",
        status="training"
    )


@app.get("/training-progress")
async def get_training_progress():
    """Get current training progress"""
    return training_status


@app.delete("/clear-uploads")
async def clear_uploads():
    """Clear uploaded training data"""
    if training_status['is_training']:
        raise HTTPException(status_code=400, detail="Cannot clear data during training")
    
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    return {"message": "Upload directory cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

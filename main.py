#!/usr/bin/env python3
"""
Minimal FastAPI Server for Ensemble Deepfake Detection
Loads models once on startup, keeps in memory for fast predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
from PIL import Image
import io
import os

# Import detectors
from detectors.ensemble_detector import EnsembleDeepfakeDetector
from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
from detectors.cemroot_detector import CemRootDetector
from detectors.vit_detector import DeepFakeDetectorV2
from detectors.preprocessing import DeepfakePreprocessor, FFTAnalyzer, LandmarkAnalyzer


# ============================================================================
# CONFIGURATION - CHANGE PATHS HERE
# NOTE: Update these paths for your machine (Linux vs Windows)
# ============================================================================

FSFM_CHECKPOINT = "./models/fsfm/checkpoint-min_train_loss.pth"
FSFM_MEAN_STD = "./models/fsfm/pretrain_ds_mean_std.txt"

# CemRoot - auto-downloads from HuggingFace if local file doesn't exist
CEMROOT_MODEL = "CemRoot/deepfake-detection-model"  # HuggingFace repo ID (or local path)

# ViT Model - can be HuggingFace name OR local path
# VIT_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"  # Original
VIT_MODEL = "jacoballessio/ai-image-detect-distilled"  # Downloads from HuggingFace
VIT_CACHE = "./models/vit-cache"

DEVICE = "cpu"  # Change to "cuda" or "mps" for GPU

# THRESHOLD CONFIGURATION - CHANGE THIS TO ADJUST SENSITIVITY
FAKE_CONFIDENCE_THRESHOLD = 0.60  # 60% threshold for detecting fake
# Lower = more sensitive (catches more fakes, more false positives)
# Higher = less sensitive (misses some fakes, fewer false positives)

# WEIGHTS FILE - Path to optimized ensemble weights
WEIGHTS_FILE = "./optimal_weights.json"

# ============================================================================


# Response models
class ModelPrediction(BaseModel):
    name: str
    prediction: str
    confidence: float
    is_fake: bool
    specialty: str
    all_probabilities: Optional[Dict[str, float]] = None


class EnsemblePrediction(BaseModel):
    summary: Dict
    models: Dict[str, ModelPrediction]
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    threshold: float


# Initialize FastAPI
app = FastAPI(
    title="Ensemble Deepfake Detector API",
    description="Multi-model deepfake detection with individual outputs",
    version="1.0.0"
)


# Global detector (loaded once on startup)
ensemble = None


@app.on_event("startup")
async def load_models():
    """Load all models once on server startup"""
    global ensemble
    
    print("\n" + "="*70)
    print("🚀 LOADING ENSEMBLE MODELS (ONE TIME)")
    print("="*70)
    
    try:
        ensemble = EnsembleDeepfakeDetector(
            fsfm_config={
                'checkpoint': FSFM_CHECKPOINT,
                'mean_std': FSFM_MEAN_STD,
                'device': DEVICE
            },
            cemroot_config={
                'model_path': CEMROOT_MODEL,
                'image_size': 128
            },
            vit_config={
                'model_name': VIT_MODEL,
                'cache_dir': VIT_CACHE,
                'device': DEVICE
            }
        )
        
        print("\n" + "="*70)
        print("✅ ALL MODELS LOADED - SERVER READY")
        print(f"🎯 Fake detection threshold: {FAKE_CONFIDENCE_THRESHOLD*100}%")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODELS: {e}")
        print("Server will start but predictions will fail")
        print("="*70 + "\n")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Ensemble Deepfake Detector API",
        "version": "1.0.0",
        "models": ["FSFM-3C", "CemRoot", "ViT-v2"],
        "endpoints": {
            "POST /predict": "Upload image for ensemble detection",
            "POST /predict/weighted": "Weighted ensemble (uses optimized weights)",
            "POST /predict/fsfm": "FSFM-3C only (4-class)",
            "POST /predict/cemroot": "CemRoot only",
            "POST /predict/vit": "ViT-v2 only",
            "POST /analyze/spectrum": "FFT frequency analysis",
            "POST /analyze/landmarks": "Facial landmark analysis",
            "POST /analyze/full": "Full preprocessing analysis",
            "GET /health": "Health check"
        },
        "threshold": FAKE_CONFIDENCE_THRESHOLD,
        "weights_file": WEIGHTS_FILE
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if ensemble is not None else "models_not_loaded",
        "models_loaded": ensemble is not None,
        "threshold": FAKE_CONFIDENCE_THRESHOLD
    }


def apply_threshold(model_result: Dict, threshold: float) -> Dict:
    """
    Apply confidence threshold to model prediction
    
    Args:
        model_result: Raw model output
        threshold: Minimum confidence to consider prediction valid
        
    Returns:
        Updated result with threshold applied
    """
    confidence = model_result['confidence']
    is_fake = model_result['is_fake']
    
    # Only trust prediction if confidence meets threshold
    if confidence < threshold:
        # Low confidence - mark as uncertain
        model_result['prediction'] = f"{model_result['prediction']} (LOW CONFIDENCE)"
        model_result['threshold_met'] = False
    else:
        model_result['threshold_met'] = True
    
    return model_result


@app.post("/predict", response_model=EnsemblePrediction)
async def predict_ensemble(file: UploadFile = File(...)):
    """
    Ensemble prediction - all 3 models
    
    Shows individual outputs + summary with threshold applied
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get predictions from all models
        result = ensemble.predict(image, cemroot_method='training_match')
        
        # Apply threshold to each model
        high_confidence_detections = []
        for model_name, model_data in result['models'].items():
            model_data = apply_threshold(model_data, FAKE_CONFIDENCE_THRESHOLD)
            
            # Track high-confidence fake detections
            if (model_data['is_fake'] and 
                model_data.get('threshold_met', False)):
                high_confidence_detections.append(model_name.upper())
        
        # Update summary with threshold info
        result['summary']['high_confidence_detections'] = high_confidence_detections
        result['summary']['threshold_applied'] = FAKE_CONFIDENCE_THRESHOLD
        
        # Convert to response model
        models_output = {}
        for key, data in result['models'].items():
            models_output[key] = ModelPrediction(
                name=data['name'],
                prediction=data['prediction'],
                confidence=data['confidence'],
                is_fake=data['is_fake'],
                specialty=data['specialty'],
                all_probabilities=data.get('all_probabilities', {})
            )
        
        return EnsemblePrediction(
            summary=result['summary'],
            models=models_output,
            threshold_used=FAKE_CONFIDENCE_THRESHOLD
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/fsfm")
async def predict_fsfm(file: UploadFile = File(...)):
    """FSFM-3C only prediction (4-class)"""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = ensemble.fsfm.predict(image, return_all_probs=True)
        
        return JSONResponse(content={
            "model": "FSFM-3C",
            "prediction": result['predicted_label'],
            "confidence": result['confidence'],
            "all_probabilities": result.get('all_probabilities', {}),
            "threshold_met": result['confidence'] >= FAKE_CONFIDENCE_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/cemroot")
async def predict_cemroot(file: UploadFile = File(...)):
    """CemRoot only prediction"""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = ensemble.cemroot.predict(
            image, 
            method='training_match', 
            return_all_probs=True
        )
        
        return JSONResponse(content={
            "model": "CemRoot",
            "prediction": result['predicted_label'],
            "confidence": result['confidence'],
            "all_probabilities": result.get('all_probabilities', {}),
            "threshold_met": result['confidence'] >= FAKE_CONFIDENCE_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    """ViT-v2 only prediction"""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = ensemble.vit.predict(image, return_all_probs=True)
        
        return JSONResponse(content={
            "model": "ViT-v2",
            "prediction": result['predicted_label'],
            "confidence": result['confidence'],
            "all_probabilities": result.get('all_probabilities', {}),
            "threshold_met": result['confidence'] >= FAKE_CONFIDENCE_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEW ENDPOINTS: Weighted Prediction & Preprocessing Analysis
# ============================================================================

@app.post("/predict/weighted")
async def predict_weighted(file: UploadFile = File(...)):
    """
    Weighted ensemble prediction using optimized weights
    
    Uses weights from optimal_weights.json if available,
    otherwise defaults to equal weights.
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = ensemble.predict_weighted(
            image, 
            weights_file=WEIGHTS_FILE,
            cemroot_method='training_match'
        )
        
        # Format response
        weighted_info = result.get('weighted_voting', {})
        
        return JSONResponse(content={
            "ensemble_prediction": "FAKE" if weighted_info.get('is_fake') else "REAL",
            "weighted_score": weighted_info.get('weighted_score', 0.5),
            "confidence": weighted_info.get('confidence', 0),
            "interpretation": weighted_info.get('interpretation', ''),
            "weights_used": weighted_info.get('weights_used', {}),
            "individual_models": {
                name: {
                    "prediction": data['prediction'],
                    "confidence": data['confidence'],
                    "is_fake": data['is_fake']
                }
                for name, data in result['models'].items()
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weighted prediction failed: {str(e)}")


@app.post("/analyze/spectrum")
async def analyze_spectrum(file: UploadFile = File(...)):
    """
    FFT (Frequency Domain) analysis
    
    Detects unnatural frequency patterns that may indicate
    AI generation or manipulation artifacts.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        analyzer = FFTAnalyzer()
        result = analyzer.analyze(image)
        
        # Remove large magnitude spectrum from response (too big for API)
        response = {
            "anomaly_score": result['anomaly_score'],
            "high_freq_ratio": result['high_freq_ratio'],
            "interpretation": result['interpretation'],
            "radial_profile_summary": {
                "low_freq_energy": float(sum(result['radial_profile'][:10])),
                "high_freq_energy": float(sum(result['radial_profile'][-10:])),
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FFT analysis failed: {str(e)}")


@app.post("/analyze/landmarks")
async def analyze_landmarks(file: UploadFile = File(...)):
    """
    Facial landmark consistency analysis
    
    Checks facial proportions and symmetry against
    expected human geometry.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(image)
        
        # Remove raw landmarks from response (too large)
        response = {
            "consistency_score": result.get('consistency_score', 0.5),
            "symmetry_score": result.get('symmetry_score', 0.5),
            "proportion_scores": result.get('proportion_scores', {}),
            "issues": result.get('issues', []),
            "interpretation": result.get('interpretation', ''),
            "error": result.get('error')
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Landmark analysis failed: {str(e)}")


@app.post("/analyze/full")
async def analyze_full(file: UploadFile = File(...)):
    """
    Full preprocessing analysis
    
    Combines FFT and landmark analysis for comprehensive
    deepfake detection preprocessing.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        preprocessor = DeepfakePreprocessor()
        result = preprocessor.analyze(image)
        
        # Summarize for API response
        response = {
            "combined_suspicion_score": result['combined_suspicion_score'],
            "interpretation": result['interpretation'],
            "recommendation": result['recommendation'],
            "fft_analysis": {
                "anomaly_score": result['fft_analysis']['anomaly_score'],
                "high_freq_ratio": result['fft_analysis']['high_freq_ratio'],
                "interpretation": result['fft_analysis']['interpretation']
            },
            "landmark_analysis": {
                "consistency_score": result['landmark_analysis'].get('consistency_score', 0.5),
                "symmetry_score": result['landmark_analysis'].get('symmetry_score', 0.5),
                "issues": result['landmark_analysis'].get('issues', []),
                "interpretation": result['landmark_analysis'].get('interpretation', '')
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full analysis failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING ENSEMBLE DEEPFAKE DETECTOR SERVER")
    print("="*70)
    print("\n📋 Configuration:")
    print(f"  • Device: {DEVICE}")
    print(f"  • Threshold: {FAKE_CONFIDENCE_THRESHOLD*100}%")
    print(f"  • FSFM Checkpoint: {FSFM_CHECKPOINT}")
    print(f"  • CemRoot Model: {CEMROOT_MODEL}")
    print(f"  • ViT Cache: {VIT_CACHE}")
    print("\n🌐 Server will start at: http://localhost:9000")
    print("📚 API docs at: http://localhost:9000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )
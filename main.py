#!/usr/bin/env python3
"""
Deepfake Ensemble API - 4-Model Detection Server
Designed for Kubernetes deployment with PVC-mounted models.

Models:
  1. FSFM-3C    (ViT)              → Face manipulation, spoofing
  2. Organika   (Swin Transformer) → AI-generated images (SDXL, DALL-E)
  3. SigLIP     (Vision-Language)   → General AI-generated detection
  4. Forensics  (Signal Processing) → FFT, landmarks, symmetry, texture, edges

All configuration via environment variables.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
from PIL import Image
import io
import os
import sys
import time
import json
import logging

# ============================================================================
# ENV CONFIGURATION
# ============================================================================

# ---- Model paths (local PVC paths or HuggingFace repo IDs) ----
FSFM_MODEL_PATH = os.environ.get("FSFM_MODEL_PATH", None)           # e.g. /models/fsfm-3c
FSFM_CHECKPOINT = os.environ.get("FSFM_CHECKPOINT", None)            # Override: direct .pth path
FSFM_MEAN_STD = os.environ.get("FSFM_MEAN_STD", None)                # Override: direct mean_std path
ORGANIKA_MODEL = os.environ.get("ORGANIKA_MODEL", "Organika/sdxl-detector")
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "prithivMLmods/open-deepfake-detection")
PREDICTOR_PATH = os.environ.get("PREDICTOR_PATH", None)               # dlib shape predictor .dat

# ---- Offline / local-only mode ----
HF_LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "false").lower() in ("true", "1", "yes")
if HF_LOCAL_ONLY:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---- Device ----
DEVICE = os.environ.get("DEVICE", "cpu")

# ---- Ensemble weights (comma-separated: fsfm,organika,siglip,forensics) ----
DETECTOR_WEIGHTS_STR = os.environ.get("DETECTOR_WEIGHTS", "0.054,0.276,0.621,0.049")
DETECTOR_WEIGHTS = {}
try:
    parts = [float(x.strip()) for x in DETECTOR_WEIGHTS_STR.split(",")]
    if len(parts) == 4:
        DETECTOR_WEIGHTS = {
            'fsfm': parts[0],
            'organika': parts[1],
            'siglip': parts[2],
            'forensics': parts[3],
        }
    elif len(parts) == 3:
        # Backward compat: 3-model weights, forensics gets 0
        DETECTOR_WEIGHTS = {
            'fsfm': parts[0],
            'organika': parts[1],
            'siglip': parts[2],
            'forensics': 0.0,
        }
except (ValueError, IndexError):
    DETECTOR_WEIGHTS = {'fsfm': 0.25, 'organika': 0.25, 'siglip': 0.25, 'forensics': 0.25}

# ---- Thresholds ----
FAKE_THRESHOLD = float(os.environ.get("FAKE_THRESHOLD", "0.5"))

# ---- Confidence capping ----
CONF_CAP_ENABLED = os.environ.get("CONF_CAP_ENABLED", "false").lower() in ("true", "1", "yes")
CONF_CAP = float(os.environ.get("CONF_CAP", "0.90"))

# ---- Uncertain band ----
UNCERTAIN_ENABLED = os.environ.get("UNCERTAIN_ENABLED", "true").lower() in ("true", "1", "yes")
UNCERTAIN_LOW = float(os.environ.get("UNCERTAIN_LOW", "0.40"))
UNCERTAIN_HIGH = float(os.environ.get("UNCERTAIN_HIGH", "0.60"))

# ---- Server ----
PORT = int(os.environ.get("PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
UVICORN_LOG_LEVEL = os.environ.get("UVICORN_LOG_LEVEL", "info").lower()

# ---- HuggingFace cache (for non-local mode) ----
HF_HOME = os.environ.get("HF_HOME", "/models/hf-cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME

# ---- Timeouts ----
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake-api")

# ============================================================================
# Response Models
# ============================================================================


class ForensicsSubScores(BaseModel):
    """Mathematical analysis sub-scores from the Face Forensics analyzer."""
    frequency: Optional[float] = Field(None, description="DCT + azimuthal FFT frequency anomaly (0-1)")
    landmark: Optional[float] = Field(None, description="Facial landmark proportion anomaly (0-1)")
    symmetry: Optional[float] = Field(None, description="Aligned bilateral symmetry anomaly in LAB (0-1)")
    texture: Optional[float] = Field(None, description="LBP + local variance texture anomaly (0-1)")
    edge: Optional[float] = Field(None, description="Edge/blending artifact anomaly (0-1)")
    noise: Optional[float] = Field(None, description="Sensor noise residual consistency anomaly (0-1)")
    color: Optional[float] = Field(None, description="Color/illumination consistency anomaly (0-1)")


class ModelResult(BaseModel):
    name: str = Field(..., description="Model name")
    prediction: str = Field(..., description="'Fake' or 'Real'")
    confidence: float = Field(..., description="Confidence (0-1)")
    is_fake: bool = Field(..., description="Model's fake verdict")
    fake_probability: Optional[float] = Field(None, description="Fake probability (0-1)")
    specialty: str = Field(..., description="Model specialty")
    sub_scores: Optional[ForensicsSubScores] = Field(None, description="Forensics sub-scores")


class AnalysisResponse(BaseModel):
    verdict: str = Field(..., description="FAKE, REAL, or UNCERTAIN")
    confidence: float = Field(..., description="Ensemble confidence (0-1)")
    fake_probability: float = Field(..., description="Weighted fake probability (0-1)")
    reasoning: str = Field(..., description="Natural language explanation for LLM relay")
    models_agreeing_fake: int = Field(..., description="Models detecting fake (0-4)")
    total_models: int = Field(4, description="Total models")
    detected_by: List[str] = Field(..., description="Models that flagged fake")
    weights_used: Dict[str, float] = Field(..., description="Ensemble weights applied")
    model_details: Dict[str, ModelResult] = Field(..., description="Per-model breakdown")
    analysis_time_ms: float = Field(..., description="Analysis time in ms")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_count: int
    device: str
    models: List[str]
    weights: Dict[str, float]
    threshold: float


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Deepfake Ensemble Detector",
    description=(
        "4-model ensemble deepfake detection API. "
        "Upload an image to POST /analyze for a weighted verdict with reasoning. "
        "Models: FSFM-3C, Organika, SigLIP, Face Forensics."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensemble = None


@app.on_event("startup")
async def load_models():
    """Load all 4 models on startup."""
    global ensemble

    from detectors.ensemble_detector import EnsembleDeepfakeDetector

    logger.info("=" * 70)
    logger.info("LOADING 4-MODEL ENSEMBLE")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"HF_LOCAL_ONLY: {HF_LOCAL_ONLY}")
    logger.info(f"Weights: {DETECTOR_WEIGHTS}")
    logger.info(f"Threshold: {FAKE_THRESHOLD}")
    logger.info(f"FSFM path: {FSFM_MODEL_PATH or FSFM_CHECKPOINT or 'auto-download'}")
    logger.info(f"Organika: {ORGANIKA_MODEL}")
    logger.info(f"SigLIP: {SIGLIP_MODEL}")
    logger.info(f"Predictor: {PREDICTOR_PATH or 'auto-download'}")

    try:
        # FSFM config: support both direct paths and model directory
        fsfm_checkpoint = FSFM_CHECKPOINT
        fsfm_mean_std = FSFM_MEAN_STD

        if FSFM_MODEL_PATH and os.path.isdir(FSFM_MODEL_PATH):
            # Auto-detect checkpoint and mean_std in model directory
            for f in os.listdir(FSFM_MODEL_PATH):
                if f.endswith('.pth') and 'checkpoint' in f.lower():
                    fsfm_checkpoint = os.path.join(FSFM_MODEL_PATH, f)
                elif f.endswith('.pth') and fsfm_checkpoint is None:
                    fsfm_checkpoint = os.path.join(FSFM_MODEL_PATH, f)
                elif 'mean_std' in f.lower():
                    fsfm_mean_std = os.path.join(FSFM_MODEL_PATH, f)
            logger.info(f"FSFM auto-detected: checkpoint={fsfm_checkpoint}, mean_std={fsfm_mean_std}")

        ensemble = EnsembleDeepfakeDetector(
            fsfm_config={
                'checkpoint': fsfm_checkpoint,
                'mean_std': fsfm_mean_std,
                'device': DEVICE,
            },
            organika_config={
                'model_name': ORGANIKA_MODEL,
                'device': DEVICE,
            },
            siglip_config={
                'model_name': SIGLIP_MODEL,
                'device': DEVICE,
            },
            forensics_config={
                'predictor_path': PREDICTOR_PATH,
                'device': DEVICE,
            },
        )

        logger.info("=" * 70)
        logger.info("ALL 4 MODELS LOADED - SERVER READY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"ERROR LOADING MODELS: {e}", exc_info=True)
        logger.error("Server will start but /analyze will return 503")


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns model status, weights, and threshold."""
    return HealthResponse(
        status="healthy" if ensemble else "models_not_loaded",
        models_loaded=ensemble is not None,
        model_count=4 if ensemble else 0,
        device=DEVICE,
        models=["FSFM-3C", "Organika", "SigLIP", "Face Forensics"] if ensemble else [],
        weights=DETECTOR_WEIGHTS,
        threshold=FAKE_THRESHOLD,
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)")
):
    """
    Analyze an image for deepfake manipulation.

    Returns verdict (FAKE/REAL/UNCERTAIN), confidence, natural language reasoning,
    and per-model breakdowns including forensic sub-scores.

    The `reasoning` field is designed for direct relay to end users by an LLM agent.
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Server still starting up.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    start = time.time()

    try:
        result = ensemble.predict_weighted(image, weights=DETECTOR_WEIGHTS)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    elapsed_ms = (time.time() - start) * 1000

    # Extract weighted score
    weighted = result.get('weighted_voting', {})
    weighted_score = weighted.get('weighted_score', 0.5)

    # Apply confidence cap
    if CONF_CAP_ENABLED:
        weighted_score = min(weighted_score, CONF_CAP) if weighted_score > 0.5 else max(weighted_score, 1 - CONF_CAP)

    # Determine verdict
    if UNCERTAIN_ENABLED and UNCERTAIN_LOW <= weighted_score <= UNCERTAIN_HIGH:
        verdict = "UNCERTAIN"
    elif weighted_score >= FAKE_THRESHOLD:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    is_fake = verdict == "FAKE"

    # Build per-model details
    model_details = {}
    for key, data in result['models'].items():
        sub_scores = None
        if key == 'forensics' and 'sub_scores' in data:
            sub_scores = ForensicsSubScores(**data['sub_scores'])

        model_details[key] = ModelResult(
            name=data['name'],
            prediction=data['prediction'],
            confidence=data['confidence'],
            is_fake=data['is_fake'],
            fake_probability=data.get('fake_probability'),
            specialty=data['specialty'],
            sub_scores=sub_scores,
        )

    reasoning = _build_reasoning(result, weighted_score, verdict, elapsed_ms)

    return AnalysisResponse(
        verdict=verdict,
        confidence=round(abs(weighted_score - 0.5) * 2, 4),
        fake_probability=round(weighted_score, 4),
        reasoning=reasoning,
        models_agreeing_fake=result['summary']['models_detecting_fake'],
        total_models=4,
        detected_by=result['summary']['detected_by'],
        weights_used=DETECTOR_WEIGHTS,
        model_details=model_details,
        analysis_time_ms=round(elapsed_ms, 1),
    )


def _build_reasoning(result, weighted_score, verdict, elapsed_ms):
    """Build natural language reasoning for LLM agents."""
    models = result['models']
    fake_count = result['summary']['models_detecting_fake']
    detected_by = result['summary']['detected_by']

    if fake_count == 0:
        verdict_text = "All 4 models agree this image appears authentic."
    elif fake_count == 4:
        verdict_text = "All 4 models unanimously detected this image as fake."
    elif fake_count >= 3:
        verdict_text = f"Strong evidence of manipulation: {fake_count}/4 models flagged fake ({', '.join(detected_by)})."
    elif fake_count == 2:
        verdict_text = f"Mixed signals: 2/4 models flagged fake ({', '.join(detected_by)})."
    else:
        verdict_text = f"Weak signal: only {detected_by[0]} flagged this image."

    if verdict == "UNCERTAIN":
        verdict_text += " The weighted score falls in the uncertain range — manual review recommended."

    conf_text = f"Weighted score: {weighted_score:.1%} fake probability."

    highlights = []
    for key, data in models.items():
        name = data['name']
        fp = data.get('fake_probability')
        if data['is_fake']:
            prob = f"{fp:.0%}" if fp is not None else f"{data['confidence']:.0%}"
            highlights.append(f"• {name}: FAKE ({prob})")
        else:
            highlights.append(f"• {name}: REAL ({data['confidence']:.0%} confidence)")

    # Forensics flags
    forensics = models.get('forensics', {})
    sub_scores = forensics.get('sub_scores', {})
    if sub_scores:
        high = [(k, v) for k, v in sub_scores.items() if v > 0.6]
        if high:
            highlights.append(f"• Forensic flags: {', '.join(f'{k} ({v:.0%})' for k, v in high)}")

    return f"{verdict_text} {conf_text}\n\n" + "\n".join(highlights)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("DEEPFAKE ENSEMBLE API SERVER")
    logger.info("=" * 70)
    logger.info(f"Port: {PORT}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"HF_LOCAL_ONLY: {HF_LOCAL_ONLY}")
    logger.info(f"Weights: {DETECTOR_WEIGHTS}")
    logger.info(f"Threshold: {FAKE_THRESHOLD}")
    logger.info(f"Uncertain band: {UNCERTAIN_LOW}-{UNCERTAIN_HIGH} (enabled={UNCERTAIN_ENABLED})")
    logger.info(f"Conf cap: {CONF_CAP} (enabled={CONF_CAP_ENABLED})")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=UVICORN_LOG_LEVEL,
        timeout_keep_alive=REQUEST_TIMEOUT_SECONDS,
    )
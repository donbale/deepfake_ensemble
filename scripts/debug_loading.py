#!/usr/bin/env python3
"""
Debug script to isolate segmentation fault during model loading.
Run each step separately to find the culprit.
"""

import os
import sys

# Force CPU mode before ANY imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("="*60)
print("DEEPFAKE ENSEMBLE - DEBUG MODEL LOADING")
print("="*60)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_step(name, func):
    """Run a test step with error handling"""
    print(f"\n[TEST] {name}...")
    sys.stdout.flush()
    try:
        result = func()
        print(f"       ✓ PASSED")
        return result
    except Exception as e:
        print(f"       ✗ FAILED: {e}")
        return None

# Step 1: Basic imports
def step1():
    import numpy as np
    print(f"       NumPy: {np.__version__}")
    return True

test_step("NumPy import", step1)

# Step 2: PyTorch
def step2():
    import torch
    print(f"       PyTorch: {torch.__version__}")
    print(f"       CUDA available: {torch.cuda.is_available()}")
    return True

test_step("PyTorch import", step2)

# Step 3: TensorFlow (AFTER PyTorch)
def step3():
    import tensorflow as tf
    print(f"       TensorFlow: {tf.__version__}")
    return True

test_step("TensorFlow import", step3)

# Step 4: PIL
def step4():
    from PIL import Image
    print(f"       PIL OK")
    return True

test_step("PIL import", step4)

# Step 5: Transformers
def step5():
    from transformers import pipeline
    print(f"       Transformers OK")
    return True

test_step("Transformers import", step5)

print("\n" + "="*60)
print("LOADING INDIVIDUAL DETECTORS")
print("="*60)

# Step 6: FSFM Detector (PyTorch)
def step6():
    from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
    print("       FSFM module imported")
    return True

test_step("FSFM module import", step6)

# Step 7: Actually load FSFM
def step7():
    from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
    detector = FSFM_UnifiedDetector(
        checkpoint_path='./models/fsfm/checkpoint-min_train_loss.pth',
        mean_std_path='./models/fsfm/pretrain_ds_mean_std.txt',
        device='cpu'
    )
    print("       FSFM model loaded")
    return detector

fsfm = test_step("FSFM model loading", step7)

# Step 8: CemRoot Detector (TensorFlow)
def step8():
    from detectors.cemroot_detector import CemRootDetector
    print("       CemRoot module imported")
    return True

test_step("CemRoot module import", step8)

# Step 9: Actually load CemRoot
def step9():
    from detectors.cemroot_detector import CemRootDetector
    detector = CemRootDetector(
        model_path='./models/cemroot/best_model_effatt.h5',
        image_size=128
    )
    print("       CemRoot model loaded")
    return detector

cemroot = test_step("CemRoot model loading", step9)

# Step 10: ViT Detector (PyTorch/Transformers)
def step10():
    from detectors.vit_detector import DeepFakeDetectorV2
    print("       ViT module imported")
    return True

test_step("ViT module import", step10)

# Step 11: Actually load ViT
def step11():
    from detectors.vit_detector import DeepFakeDetectorV2
    detector = DeepFakeDetectorV2(
        model_name='prithivMLmods/Deep-Fake-Detector-v2-Model',
        device='cpu'
    )
    print("       ViT model loaded")
    return detector

vit = test_step("ViT model loading", step11)

print("\n" + "="*60)
if fsfm and cemroot and vit:
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("   The segfault is NOT from model loading.")
    print("   Check the optimization code itself.")
else:
    print("❌ SOME MODELS FAILED TO LOAD")
    print("   Check which step failed above.")
print("="*60)

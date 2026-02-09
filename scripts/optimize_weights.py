#!/usr/bin/env python3
"""
CLI Script for Optimizing Ensemble Weights

Finds optimal weights for combining FSFM, CemRoot, and ViT predictions
using either Grid Search or Bayesian Optimization.

Usage:
    python scripts/optimize_weights.py --dataset ./data/validation --method bayesian
"""

# IMPORTANT: Set these BEFORE importing TensorFlow to avoid PyTorch/TF CUDA conflicts
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.ensemble_detector import EnsembleDeepfakeDetector
from detectors.weight_optimizer import run_optimization


def main():
    parser = argparse.ArgumentParser(
        description='Optimize ensemble weights for deepfake detection'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to validation dataset (with real/ and fake/ subdirs)')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['grid', 'bayesian'],
                        help='Optimization method (default: bayesian)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials for Bayesian optimization (default: 100)')
    parser.add_argument('--output', type=str, default='optimal_weights.json',
                        help='Output path for weights (default: optimal_weights.json)')
    
    # Model paths
    parser.add_argument('--fsfm_checkpoint', type=str,
                        default='./models/fsfm/checkpoint-min_train_loss.pth',
                        help='Path to FSFM checkpoint')
    parser.add_argument('--fsfm_mean_std', type=str,
                        default='./models/fsfm/pretrain_ds_mean_std.txt',
                        help='Path to FSFM mean_std file')
    parser.add_argument('--cemroot_model', type=str,
                        default='./models/cemroot/best_model_effatt.h5',
                        help='Path to CemRoot model')
    parser.add_argument('--vit_model', type=str,
                        default='prithivMLmods/Deep-Fake-Detector-v2-Model',
                        help='ViT model name or path')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode - load models one at a time')
    parser.add_argument('--skip-cemroot', action='store_true',
                        help='Skip CemRoot model (if TensorFlow is crashing)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔧 ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    
    # Validate dataset
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)
    
    real_dir = os.path.join(args.dataset, 'real')
    fake_dir = os.path.join(args.dataset, 'fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("❌ Dataset must have 'real/' and 'fake/' subdirectories")
        sys.exit(1)
    
    # Debug mode: test each import separately
    if args.debug:
        print("\n🔍 DEBUG MODE: Testing imports one by one...")
        
        print("  [1/4] Testing PyTorch...")
        import torch
        print(f"        ✓ PyTorch {torch.__version__}")
        
        print("  [2/4] Testing TensorFlow...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
        import tensorflow as tf
        print(f"        ✓ TensorFlow {tf.__version__}")
        
        print("  [3/4] Testing FSFM detector...")
        from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
        print("        ✓ FSFM imported")
        
        print("  [4/4] Testing ViT detector...")
        from detectors.vit_detector import DeepFakeDetectorV2
        print("        ✓ ViT imported")
        
        if not args.skip_cemroot:
            print("  [5/5] Testing CemRoot detector...")
            from detectors.cemroot_detector import CemRootDetector
            print("        ✓ CemRoot imported")
        
        print("\n✓ All imports successful!")
        print("="*70)
    
    # Initialize ensemble
    print("\n📦 Loading ensemble models...")
    
    if args.skip_cemroot:
        print("⚠️  Skipping CemRoot model (--skip-cemroot flag)")
        # Create a minimal ensemble without CemRoot
        print("  [1/2] Loading FSFM...")
        from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
        fsfm = FSFM_UnifiedDetector(
            checkpoint_path=args.fsfm_checkpoint,
            mean_std_path=args.fsfm_mean_std,
            device=args.device
        )
        
        print("  [2/2] Loading ViT...")
        from detectors.vit_detector import DeepFakeDetectorV2
        vit = DeepFakeDetectorV2(
            model_name=args.vit_model,
            device=args.device
        )
        
        print("\n⚠️  Running with 2 models only (FSFM + ViT)")
        print("   For full optimization, fix TensorFlow and remove --skip-cemroot")
        
        # TODO: Add 2-model optimization mode
        print("\n❌ 2-model mode not yet implemented. Fix TensorFlow first.")
        sys.exit(1)
    
    ensemble = EnsembleDeepfakeDetector(
        fsfm_config={
            'checkpoint': args.fsfm_checkpoint,
            'mean_std': args.fsfm_mean_std,
            'device': args.device
        },
        cemroot_config={
            'model_path': args.cemroot_model,
            'image_size': 128
        },
        vit_config={
            'model_name': args.vit_model,
            'device': args.device
        }
    )
    
    # Run optimization
    result = run_optimization(
        ensemble_detector=ensemble,
        dataset_path=args.dataset,
        method=args.method,
        n_trials=args.trials,
        output_path=args.output
    )
    
    print("\n✅ Optimization complete!")
    print(f"   Best accuracy: {result.accuracy*100:.2f}%")
    print(f"   Weights saved to: {args.output}")


if __name__ == "__main__":
    main()

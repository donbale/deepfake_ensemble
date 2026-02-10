#!/usr/bin/env python3
"""
CLI Script for Optimizing Ensemble Weights

Finds optimal weights for combining FSFM, Organika, and SigLIP predictions
using either Grid Search or Bayesian Optimization.

Usage:
    python scripts/optimize_weights.py --dataset ./data1 ./data2 ./data3 --method bayesian
    python scripts/optimize_weights.py --dataset ./data --max-samples 2000
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
    parser.add_argument('--dataset', type=str, nargs='+', required=True,
                        help='Path(s) to dataset(s) with real/ and fake/ subdirs')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['grid', 'bayesian'],
                        help='Optimization method (default: bayesian)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials for Bayesian optimization (default: 100)')
    parser.add_argument('--output', type=str, default='optimal_weights.json',
                        help='Output path for weights (default: optimal_weights.json)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max images PER DATASET (default: 2000, use 0 for all)')
    
    # Model paths
    parser.add_argument('--fsfm_checkpoint', type=str, default=None,
                        help='Path to FSFM checkpoint (auto-downloads if not set)')
    parser.add_argument('--fsfm_mean_std', type=str, default=None,
                        help='Path to FSFM mean_std file (auto-downloads if not set)')
    parser.add_argument('--organika_model', type=str,
                        default='Organika/sdxl-detector',
                        help='Organika model (HuggingFace repo ID)')
    parser.add_argument('--siglip_model', type=str,
                        default='prithivMLmods/open-deepfake-detection',
                        help='SigLIP model (HuggingFace repo ID)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode - load models one at a time')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔧 ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    
    # Validate datasets
    valid_datasets = []
    for ds_path in args.dataset:
        if not os.path.exists(ds_path):
            print(f"❌ Dataset not found: {ds_path}")
            sys.exit(1)
        
        real_dir = os.path.join(ds_path, 'real')
        fake_dir = os.path.join(ds_path, 'fake')
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print(f"❌ Dataset must have 'real/' and 'fake/' subdirectories: {ds_path}")
            sys.exit(1)
        
        valid_datasets.append(ds_path)
    
    print(f"\n📂 {len(valid_datasets)} dataset(s) to optimize across")
    for ds in valid_datasets:
        print(f"   • {ds}")
    
    # Debug mode: test each import separately
    if args.debug:
        print("\n🔍 DEBUG MODE: Testing imports one by one...")
        
        print("  [1/4] Testing PyTorch...")
        import torch
        print(f"        ✓ PyTorch {torch.__version__}")
        
        print("  [2/4] Testing FSFM detector...")
        from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
        print("        ✓ FSFM imported")
        
        print("  [3/4] Testing Organika detector...")
        from detectors.organika_detector import OrganikaDetector
        print("        ✓ Organika imported")
        
        print("  [4/4] Testing SigLIP detector...")
        from detectors.siglip_detector import SigLIPDetector
        print("        ✓ SigLIP imported")
        
        print("\n✓ All imports successful!")
        print("="*70)
    
    # Initialize ensemble
    print("\n📦 Loading ensemble models...")
    
    ensemble = EnsembleDeepfakeDetector(
        fsfm_config={
            'checkpoint': args.fsfm_checkpoint,
            'mean_std': args.fsfm_mean_std,
            'device': args.device
        },
        organika_config={
            'model_name': args.organika_model,
            'device': args.device
        },
        siglip_config={
            'model_name': args.siglip_model,
            'device': args.device
        }
    )
    
    # Run optimization
    max_samples = args.max_samples if args.max_samples > 0 else None
    result = run_optimization(
        ensemble_detector=ensemble,
        dataset_path=valid_datasets,
        method=args.method,
        n_trials=args.trials,
        max_samples=max_samples,
        output_path=args.output
    )
    
    print("\n✅ Optimization complete!")
    print(f"   Best accuracy: {result.accuracy*100:.2f}%")
    print(f"   Weights saved to: {args.output}")


if __name__ == "__main__":
    main()

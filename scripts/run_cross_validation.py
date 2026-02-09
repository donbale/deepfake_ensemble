#!/usr/bin/env python3
"""
CLI Script for K-Fold Cross-Validation

Runs stratified K-fold cross-validation to validate ensemble performance
and ensure results aren't a fluke.

Usage:
    python scripts/run_cross_validation.py --dataset ./data --folds 5
"""

# IMPORTANT: Set these BEFORE importing TensorFlow to avoid PyTorch/TF CUDA conflicts
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.ensemble_detector import EnsembleDeepfakeDetector
from evaluation.cross_validation import run_cross_validation


def main():
    parser = argparse.ArgumentParser(
        description='Run K-Fold cross-validation on deepfake ensemble'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (with real/ and fake/ subdirs)')
    parser.add_argument('--folds', type=int, default=5, choices=[5, 10],
                        help='Number of folds (default: 5)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Optional path to weights JSON for weighted voting')
    parser.add_argument('--output', type=str, default='cv_results.json',
                        help='Output path for results (default: cv_results.json)')
    
    # Model paths
    parser.add_argument('--fsfm_checkpoint', type=str,
                        default='./models/fsfm/checkpoint-min_train_loss.pth',
                        help='Path to FSFM checkpoint')
    parser.add_argument('--fsfm_mean_std', type=str,
                        default='./models/fsfm/pretrain_ds_mean_std.txt',
                        help='Path to FSFM mean_std file')
    parser.add_argument('--cemroot_model', type=str,
                        default='CemRoot/deepfake-detection-model',
                        help='CemRoot model (HuggingFace repo ID or local .h5 path)')
    parser.add_argument('--vit_model', type=str,
                        default='jacoballessio/ai-image-detect-distilled',
                        help='ViT model name (HuggingFace) or local path')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"🔄 {args.folds}-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Validate dataset
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Load weights if provided
    weights = None
    if args.weights and os.path.exists(args.weights):
        with open(args.weights, 'r') as f:
            data = json.load(f)
            weights = data.get('weights', None)
        print(f"📊 Using weights from: {args.weights}")
    
    # Initialize ensemble
    print("\n📦 Loading ensemble models...")
    
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
    
    # Run cross-validation
    result = run_cross_validation(
        ensemble_detector=ensemble,
        dataset_path=args.dataset,
        k=args.folds,
        weights=weights,
        output_path=args.output
    )
    
    print("\n✅ Cross-validation complete!")
    print(f"   Mean Accuracy: {result.mean_accuracy*100:.2f}% ± {result.std_accuracy*100:.2f}%")
    print(f"   Mean F1 Score: {result.mean_f1*100:.2f}% ± {result.std_f1*100:.2f}%")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()

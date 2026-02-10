#!/usr/bin/env python3
"""
Ensemble Deepfake Detector - Individual Model Dashboard
Shows each model's prediction separately to leverage their different specializations

Models:
1. FSFM-3C (Wolowolo) - 4-class ViT - Best for: Spoofing, advanced manipulations
2. Secondary - ViT fine-tuned on deepfakes - Best for: General deepfake detection
3. ViT-v2 (jacoballessio) - ViT-base - Best for: AI-generated images
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os
import sys

# Import individual detectors
from .fsfm_unified_detector import FSFM_UnifiedDetector
from .secondary_detector import SecondaryDetector
from .vit_detector import DeepFakeDetectorV2


class EnsembleDeepfakeDetector:
    """
    Ensemble detector showing individual model outputs
    No voting/aggregation - shows each model's specialty
    """

    def __init__(self, fsfm_config, secondary_config, vit_config):
        """
        Initialize ensemble detector

        Args:
            fsfm_config: Dict with 'checkpoint', 'mean_std', 'device'
            secondary_config: Dict with 'model_name', 'device'
            vit_config: Dict with 'model_name', 'cache_dir', 'device'
        """
        print("\n" + "="*70)
        print("INITIALIZING ENSEMBLE DEEPFAKE DETECTOR")
        print("="*70)

        # Initialize Model 1: FSFM-3C
        print("\n[1/3] Loading FSFM-3C Unified Detector...")
        self.fsfm = FSFM_UnifiedDetector(
            checkpoint_path=fsfm_config['checkpoint'],
            mean_std_path=fsfm_config['mean_std'],
            device=fsfm_config.get('device', 'cuda')
        )

        # Initialize Model 2: Secondary
        print("\n[2/3] Loading Secondary Detector...")
        self.secondary = SecondaryDetector(
            model_name=secondary_config.get('model_name'),
            device=secondary_config.get('device', 'cuda')
        )

        # Initialize Model 3: ViT-v2
        print("\n[3/3] Loading ViT-v2 Detector...")
        self.vit = DeepFakeDetectorV2(
            model_name=vit_config.get('model_name', 'prithivMLmods/Deep-Fake-Detector-v2-Model'),
            cache_dir=vit_config.get('cache_dir', None),
            device=vit_config.get('device', 'cuda')
        )

        print("\n" + "="*70)
        print("✓ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*70 + "\n")

    def predict(self, image_path):
        """
        Get predictions from all 3 models individually

        Args:
            image_path: Path to image or PIL Image

        Returns:
            Dictionary with individual model predictions (no aggregation)
        """
        print(f"\n🔍 Running individual model predictions on: {image_path}")

        # Get predictions from all models
        print("  [1/3] FSFM-3C predicting...")
        fsfm_result = self.fsfm.predict(image_path, return_all_probs=True)

        print("  [2/3] Secondary predicting...")
        secondary_result = self.secondary.predict(image_path, return_all_probs=True)

        print("  [3/3] ViT-v2 predicting...")
        vit_result = self.vit.predict(image_path, return_all_probs=True)

        # Use standardized is_fake from each model (handles any label format)
        fsfm_is_fake = fsfm_result['predicted_class'] != 0
        secondary_is_fake = secondary_result.get('is_fake', 'fake' in secondary_result['predicted_label'].lower())
        vit_is_fake = vit_result.get('is_fake', 'fake' in vit_result['predicted_label'].lower())

        # Build result showing each model's output
        result = {
            'models': {
                'fsfm': {
                    'name': 'FSFM-3C (4-class)',
                    'prediction': fsfm_result['predicted_label'],
                    'confidence': fsfm_result['confidence'],
                    'is_fake': fsfm_is_fake,
                    'all_probabilities': fsfm_result.get('all_probabilities', {}),
                    'specialty': 'Spoofing, Physical attacks, Advanced manipulations'
                },
                'secondary': {
                    'name': 'Secondary (ViT fine-tuned)',
                    'prediction': secondary_result['predicted_label'],
                    'confidence': secondary_result['confidence'],
                    'is_fake': secondary_is_fake,
                    'fake_probability': secondary_result.get('fake_probability'),
                    'all_probabilities': secondary_result.get('all_probabilities', {}),
                    'specialty': 'General deepfake detection'
                },
                'vit': {
                    'name': 'ViT-v2 (Transformers)',
                    'prediction': vit_result['predicted_label'],
                    'confidence': vit_result['confidence'],
                    'is_fake': vit_is_fake,
                    'fake_probability': vit_result.get('fake_probability'),
                    'all_probabilities': vit_result.get('all_probabilities', {}),
                    'specialty': 'Traditional deepfakes, ChatGPT/Gemini generation'
                }
            },
            'summary': {
                'models_detecting_fake': sum([fsfm_is_fake, secondary_is_fake, vit_is_fake]),
                'total_models': 3,
                'detected_by': []
            }
        }

        # Track which models detected fake
        if fsfm_is_fake:
            result['summary']['detected_by'].append('FSFM-3C')
        if secondary_is_fake:
            result['summary']['detected_by'].append('Secondary')
        if vit_is_fake:
            result['summary']['detected_by'].append('ViT-v2')

        return result

    def predict_weighted(self, image_path, weights=None, 
                         weights_file='optimal_weights.json'):
        """
        Get weighted ensemble prediction using optimized weights
        
        Args:
            image_path: Path to image or PIL Image
            weights: Optional dict of weights {'fsfm': w1, 'secondary': w2, 'vit': w3}
                     If not provided, loads from weights_file
            weights_file: Path to JSON file with optimized weights
            
        Returns:
            Dictionary with weighted prediction and individual model outputs
        """
        import os
        import json
        
        # Load weights from file if not provided
        if weights is None:
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    weights = data.get('weights', {})
            else:
                # Default to equal weights
                weights = {'fsfm': 1/3, 'secondary': 1/3, 'vit': 1/3}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Get individual predictions
        result = self.predict(image_path)
        
        # Calculate weighted score
        fake_probs = self.get_fake_probabilities(result)
        
        weighted_score = sum(
            weights.get(name, 1/3) * prob 
            for name, prob in fake_probs.items()
        )
        
        # Add weighted voting to result
        result['weighted_voting'] = {
            'weights_used': weights,
            'weighted_score': weighted_score,
            'is_fake': weighted_score >= 0.5,
            'confidence': abs(weighted_score - 0.5) * 2,  # Distance from threshold
            'interpretation': self._interpret_weighted_score(weighted_score)
        }
        
        return result
    
    def get_fake_probabilities(self, prediction_result):
        """
        Extract fake probabilities from prediction result
        
        Used for weight optimization and weighted voting.
        
        Args:
            prediction_result: Output from predict() method
            
        Returns:
            Dict mapping model name to fake probability
        """
        probs = {}
        
        for name, data in prediction_result['models'].items():
            if data['is_fake']:
                # If model says fake, use its confidence as fake probability
                probs[name] = data['confidence']
            else:
                # If model says real, fake probability is 1 - confidence
                probs[name] = 1 - data['confidence']
        
        return probs
    
    def _interpret_weighted_score(self, score):
        """Interpret weighted score"""
        if score < 0.3:
            return "LOW - Likely authentic image"
        elif score < 0.5:
            return "MEDIUM-LOW - Probably real but some doubts"
        elif score < 0.7:
            return "MEDIUM-HIGH - Suspicious, likely manipulated"
        else:
            return "HIGH - Strong evidence of deepfake"


def print_ensemble_result(result):
    """Pretty print individual model results"""
    print("\n" + "="*70)
    print("🎯 ENSEMBLE DEEPFAKE DETECTION - INDIVIDUAL MODEL OUTPUTS")
    print("="*70)

    # Summary
    fake_count = result['summary']['models_detecting_fake']
    total = result['summary']['total_models']

    print(f"\n📊 SUMMARY: {fake_count}/{total} models detected FAKE")

    if fake_count == 0:
        print("✅ All models agree: REAL image")
    elif fake_count == total:
        print("🚨 All models agree: FAKE image")
    else:
        print(f"⚠️  Mixed results - {fake_count} model(s) detected fake")
        print(f"   Detected by: {', '.join(result['summary']['detected_by'])}")

    # Individual model outputs
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PREDICTIONS:")
    print("="*70)

    for model_key, model_data in result['models'].items():
        print(f"\n{'─'*70}")
        print(f"🤖 {model_data['name']}")
        print(f"{'─'*70}")

        # Prediction
        emoji = "🚨" if model_data['is_fake'] else "✅"
        print(f"{emoji} PREDICTION: {model_data['prediction']}")
        print(f"📊 CONFIDENCE: {model_data['confidence']*100:.2f}%")
        print(f"🎯 SPECIALTY: {model_data['specialty']}")

        if 'preprocessing' in model_data:
            print(f"🔧 PREPROCESSING: {model_data['preprocessing']}")

        # Show all probabilities
        if model_data['all_probabilities']:
            print(f"\n📈 CLASS PROBABILITIES:")
            for label, prob in model_data['all_probabilities'].items():
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {label:30s} {bar} {prob*100:5.2f}%")

    print("\n" + "="*70)
    print("💡 INTERPRETATION GUIDE:")
    print("="*70)
    print("• FSFM-3C: Look for 'Deepfake', 'Diffusion', or 'Spoofing' classes")
    print("• CemRoot: High accuracy (95%) with training_match preprocessing")
    print("• ViT-v2: Strong at traditional deepfakes and ChatGPT images")
    print("\n➡️  If ANY model detects fake with high confidence, investigate!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Deepfake Detection - Individual Model Dashboard'
    )

    # Input
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')

    # FSFM-3C config
    parser.add_argument('--fsfm_checkpoint', type=str,
                        default='../models/fsfm/checkpoint-min_train_loss.pth',
                        help='Path to FSFM checkpoint-min_train_loss.pth')
    parser.add_argument('--fsfm_mean_std', type=str,
                        default='../models/fsfm/pretrain_ds_mean_std.txt',
                        help='Path to FSFM pretrain_ds_mean_std.txt')

    # CemRoot config
    parser.add_argument('--cemroot_model', type=str,
                        default='../models/cemroot/best_model_effatt.h5',
                        help='Path to CemRoot best_model_effatt.h5')
    parser.add_argument('--cemroot_method', type=str, default='training_match',
                        choices=['training_match', 'simple_norm', 'efficientnet'],
                        help='CemRoot preprocessing method')

    # ViT-v2 config
    parser.add_argument('--vit_model', type=str,
                        default='prithivMLmods/Deep-Fake-Detector-v2-Model',
                        help='ViT model name or path')
    parser.add_argument('--vit_cache_dir', type=str,
                        default='../models/vit-v2',
                        help='ViT cache directory')

    # Device config
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device for PyTorch models')

    args = parser.parse_args()

    # Initialize ensemble
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
            'cache_dir': args.vit_cache_dir,
            'device': args.device
        }
    )

    # Run individual predictions
    result = ensemble.predict(
        args.image,
        cemroot_method=args.cemroot_method
    )

    # Print results
    print_ensemble_result(result)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("ENSEMBLE DEEPFAKE DETECTOR - INDIVIDUAL MODEL DASHBOARD")
        print("="*70)
        print("\nShows each model's prediction separately (no voting/aggregation)")
        print("\nModels:")
        print("  1. FSFM-3C - Best for: Spoofing, advanced manipulations")
        print("  2. CemRoot - Best for: Diffusion models, AIGC")
        print("  3. ViT-v2  - Best for: Traditional deepfakes, ChatGPT")
        print("\nUsage:")
        print("  python ensemble_detector.py --image <path_to_image>")
        print("\nExample:")
        print("  python ensemble_detector.py --image test.jpg")
        print("\n" + "="*70)
        print("\nWhy individual outputs?")
        print("  • Different models catch different generation methods")
        print("  • Voting can hide important detections")
        print("  • See which model's specialty matches the image")
        print("  • ANY high-confidence fake detection is valuable!")
        print("="*70 + "\n")
    else:
        main()
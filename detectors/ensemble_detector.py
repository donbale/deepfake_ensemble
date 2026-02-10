#!/usr/bin/env python3
"""
Ensemble Deepfake Detector - Individual Model Dashboard
Shows each model's prediction separately to leverage their different specializations

Models:
1. FSFM-3C (Wolowolo) - 4-class ViT - Best for: Face deepfakes, spoofing
2. Organika (Swin Transformer) - Best for: AI-generated images (SDXL, DALL-E)
3. SigLIP (Vision-Language) - Best for: General AI-generated detection
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os
import sys

# Import individual detectors
from .fsfm_unified_detector import FSFM_UnifiedDetector
from .organika_detector import OrganikaDetector
from .siglip_detector import SigLIPDetector


class EnsembleDeepfakeDetector:
    """
    Ensemble detector showing individual model outputs
    No voting/aggregation - shows each model's specialty
    """

    def __init__(self, fsfm_config, organika_config, siglip_config):
        """
        Initialize ensemble detector

        Args:
            fsfm_config: Dict with 'checkpoint', 'mean_std', 'device'
            organika_config: Dict with 'model_name', 'device'
            siglip_config: Dict with 'model_name', 'device'
        """
        print("\n" + "="*70)
        print("INITIALIZING ENSEMBLE DEEPFAKE DETECTOR")
        print("="*70)

        # Initialize Model 1: FSFM-3C (ViT, face forensics)
        print("\n[1/3] Loading FSFM-3C Unified Detector...")
        self.fsfm = FSFM_UnifiedDetector(
            checkpoint_path=fsfm_config['checkpoint'],
            mean_std_path=fsfm_config['mean_std'],
            device=fsfm_config.get('device', 'cuda')
        )

        # Initialize Model 2: Organika (Swin, SDXL/diffusion detection)
        print("\n[2/3] Loading Organika SDXL Detector...")
        self.organika = OrganikaDetector(
            model_name=organika_config.get('model_name'),
            device=organika_config.get('device', 'cuda')
        )

        # Initialize Model 3: SigLIP (Vision-Language, general AI detection)
        print("\n[3/3] Loading SigLIP Detector...")
        self.siglip = SigLIPDetector(
            model_name=siglip_config.get('model_name'),
            device=siglip_config.get('device', 'cuda')
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

        print("  [2/3] Organika predicting...")
        organika_result = self.organika.predict(image_path, return_all_probs=True)

        print("  [3/3] SigLIP predicting...")
        siglip_result = self.siglip.predict(image_path, return_all_probs=True)

        # Use standardized is_fake from each model
        fsfm_is_fake = fsfm_result['predicted_class'] != 0
        organika_is_fake = organika_result.get('is_fake', False)
        siglip_is_fake = siglip_result.get('is_fake', False)

        # Build result showing each model's output
        result = {
            'models': {
                'fsfm': {
                    'name': 'FSFM-3C (ViT, 4-class)',
                    'prediction': fsfm_result['predicted_label'],
                    'confidence': fsfm_result['confidence'],
                    'is_fake': fsfm_is_fake,
                    'all_probabilities': fsfm_result.get('all_probabilities', {}),
                    'specialty': 'Face deepfakes, spoofing, physical attacks'
                },
                'organika': {
                    'name': 'Organika (Swin, SDXL)',
                    'prediction': organika_result['predicted_label'],
                    'confidence': organika_result['confidence'],
                    'is_fake': organika_is_fake,
                    'fake_probability': organika_result.get('fake_probability'),
                    'all_probabilities': organika_result.get('all_probabilities', {}),
                    'specialty': 'AI-generated images (SDXL, DALL-E, Midjourney)'
                },
                'siglip': {
                    'name': 'SigLIP (Vision-Language)',
                    'prediction': siglip_result['predicted_label'],
                    'confidence': siglip_result['confidence'],
                    'is_fake': siglip_is_fake,
                    'fake_probability': siglip_result.get('fake_probability'),
                    'all_probabilities': siglip_result.get('all_probabilities', {}),
                    'specialty': 'General AI-generated detection, digital forensics'
                }
            },
            'summary': {
                'models_detecting_fake': sum([fsfm_is_fake, organika_is_fake, siglip_is_fake]),
                'total_models': 3,
                'detected_by': []
            }
        }

        # Track which models detected fake
        if fsfm_is_fake:
            result['summary']['detected_by'].append('FSFM-3C')
        if organika_is_fake:
            result['summary']['detected_by'].append('Organika')
        if siglip_is_fake:
            result['summary']['detected_by'].append('SigLIP')

        return result

    def predict_weighted(self, image_path, weights=None, 
                         weights_file='optimal_weights.json'):
        """
        Get weighted ensemble prediction using optimized weights
        
        Args:
            image_path: Path to image or PIL Image
            weights: Optional dict of weights {'fsfm': w1, 'organika': w2, 'siglip': w3}
                     If not provided, loads from weights_file
            weights_file: Path to JSON file with optimized weights
            
        Returns:
            Dictionary with weighted prediction and individual model outputs
        """
        import json
        
        # Load weights from file if not provided
        if weights is None:
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    weights = data.get('weights', {})
            else:
                # Default to equal weights
                weights = {'fsfm': 1/3, 'organika': 1/3, 'siglip': 1/3}
        
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
            # Use standardized fake_probability if available
            if 'fake_probability' in data and data['fake_probability'] is not None:
                probs[name] = data['fake_probability']
            elif data['is_fake']:
                probs[name] = data['confidence']
            else:
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

        if 'fake_probability' in model_data and model_data['fake_probability'] is not None:
            print(f"🔴 FAKE PROBABILITY: {model_data['fake_probability']*100:.2f}%")

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
    print("• Organika: Specialized in SDXL/DALL-E/Midjourney detection (98% acc)")
    print("• SigLIP: General AI-generated image detection (vision-language model)")
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
    parser.add_argument('--fsfm_checkpoint', type=str, default=None,
                        help='Path to FSFM checkpoint (auto-downloads if not set)')
    parser.add_argument('--fsfm_mean_std', type=str, default=None,
                        help='Path to FSFM mean_std file (auto-downloads if not set)')

    # Organika config
    parser.add_argument('--organika_model', type=str,
                        default='Organika/sdxl-detector',
                        help='Organika model name or path')

    # SigLIP config
    parser.add_argument('--siglip_model', type=str,
                        default='prithivMLmods/open-deepfake-detection',
                        help='SigLIP model name or path')

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
        organika_config={
            'model_name': args.organika_model,
            'device': args.device
        },
        siglip_config={
            'model_name': args.siglip_model,
            'device': args.device
        }
    )

    # Run individual predictions
    result = ensemble.predict(args.image)

    # Print results
    print_ensemble_result(result)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("ENSEMBLE DEEPFAKE DETECTOR - INDIVIDUAL MODEL DASHBOARD")
        print("="*70)
        print("\nShows each model's prediction separately (no voting/aggregation)")
        print("\nModels:")
        print("  1. FSFM-3C  - Best for: Face deepfakes, spoofing")
        print("  2. Organika - Best for: AI-generated images (SDXL, DALL-E)")
        print("  3. SigLIP   - Best for: General AI-generated detection")
        print("\nUsage:")
        print("  python ensemble_detector.py --image <path_to_image>")
        print("\n" + "="*70)
    else:
        main()
#!/usr/bin/env python3
"""
Deep-Fake-Detector-v2 - Standalone Inference Script
Model: prithivMLmods/Deep-Fake-Detector-v2-Model
Architecture: Vision Transformer (ViT-base-patch16-224)
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import argparse
import os


class DeepFakeDetectorV2:
    """ViT-based deepfake detector (prithivMLmods)"""

    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model",
                 cache_dir=None, device='cuda', local_files_only=False):
        """
        Initialize the detector

        Args:
            model_name: HuggingFace model name or local path
            cache_dir: Directory to cache/find downloaded models
            device: 'cuda' or 'cpu'
            local_files_only: If True, only use local cached files (no download)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Check if cache_dir exists and has files
        if cache_dir and os.path.exists(cache_dir):
            cache_has_files = any(os.scandir(cache_dir))
            if cache_has_files:
                print(f"✓ Found cached model in: {cache_dir}")
                local_files_only = True
            else:
                print(f"⚠️  Cache dir exists but empty: {cache_dir}")

        # Load model and processor
        print(f"Loading model: {model_name}")
        if local_files_only:
            print("Using cached files only (no download)")
        else:
            print("(Will download ~350MB if not cached)")

        try:
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            ).to(self.device)

            self.processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
        except Exception as e:
            if local_files_only:
                print(f"❌ Failed to load from cache: {e}")
                print("Retrying with download enabled...")
                self.model = ViTForImageClassification.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(self.device)

                self.processor = ViTImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            else:
                raise

        self.model.eval()
        print("✓ Model loaded successfully!")

        # Get class labels from config
        self.class_labels = self.model.config.id2label
        print(f"Classes: {self.class_labels}")
        
        # Determine which class index means "fake"
        self.fake_class_idx = self._detect_fake_class()

    def _detect_fake_class(self):
        """
        Detect which class index corresponds to 'fake/AI-generated'.
        """
        labels_lower = {k: v.lower() for k, v in self.class_labels.items()}
        
        fake_keywords = ['fake', 'ai', 'deepfake', 'synthetic', 'generated', 'manipulated']
        real_keywords = ['real', 'human', 'authentic', 'original', 'genuine', 'nature']
        
        fake_idx = None
        real_idx = None
        
        for idx, label in labels_lower.items():
            if any(kw in label for kw in fake_keywords):
                fake_idx = idx
            if any(kw in label for kw in real_keywords):
                real_idx = idx
        
        if fake_idx is not None:
            print(f"   → Fake class: idx={fake_idx} ('{self.class_labels[fake_idx]}')")
            return fake_idx
        
        if real_idx is not None:
            other_idx = 1 - real_idx
            print(f"   → Fake class: idx={other_idx} (inferred from real='{self.class_labels[real_idx]}')")
            return other_idx
        
        print(f"   ⚠️  Unknown labels, assuming class 1 = fake (convention)")
        return 1

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')

        # Process with ViT processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def predict(self, image_path, return_all_probs=False):
        """Predict if image is real or fake"""
        # Preprocess
        inputs = self.preprocess_image(image_path)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Get predicted class
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[predicted_class].item()
        label = self.class_labels[predicted_class]
        
        # Standardized fake probability
        fake_prob = probs[self.fake_class_idx].item()
        is_fake = (predicted_class == self.fake_class_idx)

        # Build result
        result = {
            'predicted_class': predicted_class,
            'predicted_label': label,
            'confidence': confidence,
            'is_fake': is_fake,
            'fake_probability': fake_prob,
        }

        if return_all_probs:
            result['all_probabilities'] = {
                self.class_labels[i]: probs[i].item()
                for i in range(len(self.class_labels))
            }

        return result

    def predict_batch(self, image_paths):
        """Predict on multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_all_probs=True)
            results.append(result)
        return results


def print_prediction_result(result):
    """Pretty print prediction results"""
    print("\n" + "="*70)
    print("DEEP-FAKE-DETECTOR-V2 - PREDICTION RESULT")
    print("="*70)

    # Color coding
    if "Deepfake" in result['predicted_label']:
        emoji = "🚨"
    else:
        emoji = "✅"

    print(f"\n{emoji} PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")

    if 'all_probabilities' in result:
        print("\n📈 CLASS PROBABILITIES:")
        for label, prob in result['all_probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {label:15s} {bar} {prob*100:5.2f}%")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Deep-Fake-Detector-v2 Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str,
                        default='prithivMLmods/Deep-Fake-Detector-v2-Model',
                        help='Model name or path')
    parser.add_argument('--cache_dir', type=str,
                        default='../models/vit-v2',
                        help='Directory to cache downloaded models')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--local_only', action='store_true',
                        help='Only use local cached files (no download)')

    args = parser.parse_args()

    # Initialize detector
    print("\n🚀 Initializing Deep-Fake-Detector-v2...")
    detector = DeepFakeDetectorV2(
        model_name=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_only
    )

    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    result = detector.predict(args.image, return_all_probs=True)

    # Print results
    print_prediction_result(result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("DEEP-FAKE-DETECTOR-V2 - STANDALONE INFERENCE SCRIPT")
        print("="*70)
        print("\nUsage:")
        print("  python vit_detector.py --image test.jpg")
        print("\nTo use local cache only (no download):")
        print("  python vit_detector.py --image test.jpg --local_only")
        print("="*70 + "\n")
    else:
        main()
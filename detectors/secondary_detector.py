#!/usr/bin/env python3
"""
Secondary Deepfake Detector - Configurable ViT-based model
Default: buildborderless/CommunityForensics-DeepfakeDet-ViT

This model was trained on 2.7M samples from 4,800+ generators.
Alternatives to try:
- dima806/deepfake_vs_real_image_detection
- yermandy/deepfake-detection
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os


class SecondaryDetector:
    """
    ViT-based deepfake detector (configurable model)
    Default: buildborderless/CommunityForensics-DeepfakeDet-ViT
    
    Detects 2 classes:
    - fake/AI: AI-generated/deepfake
    - real/human: Authentic image
    """
    
    # Default HuggingFace model ID (can be overridden)
    HF_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    
    def __init__(self, model_name=None, cache_dir="./models/secondary", device='cuda'):
        """
        Initialize the detector
        
        Args:
            model_name: HuggingFace model ID or local path
            cache_dir: Directory to cache downloaded models
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Use default model if not specified
        if model_name is None:
            model_name = self.HF_MODEL_ID
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        
        try:
            # Load model and processor
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(self.device)
            
            self.processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            self.model.eval()
            print("✓ Model loaded successfully!")
            
            # Get class labels from config
            self.class_labels = self.model.config.id2label
            print(f"Classes: {self.class_labels}")
            
            # Determine which class index means "fake"
            self.fake_class_idx = self._detect_fake_class()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _detect_fake_class(self):
        """
        Detect which class index corresponds to 'fake/AI-generated'.
        Handles various label formats from different HuggingFace models.
        """
        labels_lower = {k: v.lower() for k, v in self.class_labels.items()}
        
        # Known fake indicators
        fake_keywords = ['fake', 'ai', 'deepfake', 'synthetic', 'generated', 'manipulated']
        # Known real indicators
        real_keywords = ['real', 'human', 'authentic', 'original', 'genuine', 'nature']
        
        fake_idx = None
        real_idx = None
        
        for idx, label in labels_lower.items():
            if any(kw in label for kw in fake_keywords):
                fake_idx = idx
            if any(kw in label for kw in real_keywords):
                real_idx = idx
        
        # If we found fake explicitly
        if fake_idx is not None:
            print(f"   → Fake class: idx={fake_idx} ('{self.class_labels[fake_idx]}')")
            return fake_idx
        
        # If we only found real, fake is the other one
        if real_idx is not None:
            other_idx = 1 - real_idx  # Works for binary (0/1)
            print(f"   → Fake class: idx={other_idx} (inferred from real='{self.class_labels[real_idx]}')")
            return other_idx
        
        # Generic labels (LABEL_0/LABEL_1) - convention: class 1 = fake
        print(f"   ⚠️  Unknown labels, assuming class 1 = fake (convention)")
        return 1
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Get the expected image size from model config
        model_image_size = getattr(self.model.config, 'image_size', 384)
        
        # Resize image to model's expected size before processing
        image = image.resize((model_image_size, model_image_size), Image.Resampling.LANCZOS)
        
        # Process with ViT processor
        inputs = self.processor(images=image, return_tensors="pt", do_resize=False)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, image_path, return_all_probs=False):
        """
        Predict if image is real or fake
        
        Args:
            image_path: Path to image or PIL Image
            return_all_probs: If True, return probabilities for all classes
            
        Returns:
            Dictionary with prediction results including standardized is_fake
        """
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
        
        # Standardized fake probability (probability of the "fake" class)
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
    print("SECONDARY DETECTOR - PREDICTION RESULT")
    print("="*70)
    
    # Color coding
    if "fake" in result['predicted_label'].lower():
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Secondary Deepfake Detector Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("\n🚀 Initializing Secondary Deepfake Detector...")
    detector = SecondaryDetector(device=args.device)
    
    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    result = detector.predict(args.image, return_all_probs=True)
    
    # Print results
    print_prediction_result(result)

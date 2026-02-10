#!/usr/bin/env python3
"""
SigLIP Deepfake Detector - Vision-Language model for AI image detection  
Model: prithivMLmods/open-deepfake-detection

Architecture: SigLIP2 (google/siglip2-base-patch16-512)
Input: 512x512 images
Labels: Class 0 = Fake, Class 1 = Real
Dataset: OpenDeepfake-Preview (20k images, 2025)
Speciality: General AI-generated image detection, digital forensics
"""

import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os


class SigLIPDetector:
    """
    SigLIP2-based deepfake/AI image detector
    Model: prithivMLmods/open-deepfake-detection
    
    Uses vision-language architecture (different from ViT/Swin)
    for general AI-generated image detection.
    """
    
    HF_MODEL_ID = "prithivMLmods/open-deepfake-detection"
    
    def __init__(self, model_name=None, cache_dir="./models/siglip", device='cuda'):
        """
        Initialize the detector
        
        Args:
            model_name: HuggingFace model ID or local path
            cache_dir: Directory to cache downloaded models
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        if model_name is None:
            model_name = self.HF_MODEL_ID
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        print(f"(Will download ~400MB if not cached)")
        
        try:
            # Load SigLIP model via Auto classes
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(self.device)
            
            self.processor = AutoImageProcessor.from_pretrained(
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
        """Detect which class index corresponds to 'fake/AI-generated'."""
        labels_lower = {k: v.lower() for k, v in self.class_labels.items()}
        
        fake_keywords = ['fake', 'ai', 'artificial', 'deepfake', 'synthetic', 'generated', 'manipulated']
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
        
        print(f"   ⚠️  Unknown labels, assuming class 0 = fake")
        return 0
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Use Auto processor (handles SigLIP's 512x512 input)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, image_path, return_all_probs=False):
        """
        Predict if image is real or AI-generated
        
        Returns:
            Dictionary with standardized prediction results
        """
        inputs = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[predicted_class].item()
        label = self.class_labels[predicted_class]
        
        # Standardized fake probability
        fake_prob = probs[self.fake_class_idx].item()
        is_fake = (predicted_class == self.fake_class_idx)
        
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
        return [self.predict(p, return_all_probs=True) for p in image_paths]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SigLIP Deepfake Detector Inference')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    print("\n🚀 Initializing SigLIP Deepfake Detector...")
    detector = SigLIPDetector(device=args.device)
    
    print(f"\n🔍 Analyzing image: {args.image}")
    result = detector.predict(args.image, return_all_probs=True)
    
    emoji = "🚨" if result['is_fake'] else "✅"
    print(f"\n{emoji} PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")
    print(f"🎯 FAKE PROBABILITY: {result['fake_probability']*100:.2f}%")

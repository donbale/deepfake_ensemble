#!/usr/bin/env python3
"""
FSFM-3C Unified Detector - Standalone Inference Script
Model: Wolowolo/fsfm-3c (Unified-detector v1)

Detects 4 classes:
- Class 0: Real/Bonafide
- Class 1: Deepfake  
- Class 2: Diffusion/AIGC
- Class 3: Spoofing/Presentation-attacks
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import argparse
import os
import json

# Import the custom ViT architecture
from . import models_vit


class FSFM_UnifiedDetector:
    """
    FSFM-3C Unified Detector for face security
    Detects: Real, Deepfake, Diffusion, Spoofing
    """
    
    # HuggingFace repo for auto-download
    HF_REPO_ID = "Wolowolo/fsfm-3c"
    HF_CHECKPOINT_PATH = "finetuned_models/Unified-detector/v1_Fine-tuned_on_4_classes/checkpoint-min_train_loss.pth"
    HF_MEAN_STD_PATH = "finetuned_models/Unified-detector/v1_Fine-tuned_on_4_classes/pretrain_ds_mean_std.txt"
    
    def __init__(self, checkpoint_path=None, mean_std_path=None, device='cuda', 
                 cache_dir="./models/fsfm"):
        """
        Initialize the detector
        
        Args:
            checkpoint_path: Path to checkpoint .pth file (or None to auto-download)
            mean_std_path: Path to mean_std .txt file (or None to auto-download)
            device: 'cuda' or 'cpu'
            cache_dir: Directory to cache downloaded models
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        print(f"Using device: {self.device}")
        
        # Resolve paths (download from HuggingFace if needed)
        checkpoint_path = self._resolve_path(checkpoint_path, self.HF_CHECKPOINT_PATH, "checkpoint")
        mean_std_path = self._resolve_path(mean_std_path, self.HF_MEAN_STD_PATH, "mean_std")
        
        # Load normalization stats
        self.mean, self.std = self._load_normalization_stats(mean_std_path)
        print(f"Normalization - Mean: {self.mean}, Std: {self.std}")
        
        # Build model architecture
        self.model = models_vit.__dict__['vit_base_patch16'](
            num_classes=4,           # 4-class detection
            drop_path_rate=0.1,
            global_pool=True,        # Use global average pooling
        ).to(self.device)
        
        # Load trained weights
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        print("✓ Model loaded successfully!")
        
        # Create preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT standard input size
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Class labels
        self.class_labels = {
            0: "Real/Bonafide",
            1: "Deepfake",
            2: "Diffusion/AIGC",
            3: "Spoofing/Presentation-attack"
        }

    def _resolve_path(self, local_path, hf_filename, file_type):
        """
        Resolve file path - download from HuggingFace if needed
        
        Args:
            local_path: Local path or None
            hf_filename: Filename in HuggingFace repo
            file_type: Description for logging
            
        Returns:
            Local path to the file
        """
        # If local path provided and exists, use it
        if local_path and os.path.exists(local_path):
            return local_path
        
        # Need to download from HuggingFace
        if local_path:
            print(f"⚠️  {file_type} not found at: {local_path}")
        print(f"📥 Downloading {file_type} from HuggingFace: {self.HF_REPO_ID}")
        
        try:
            from huggingface_hub import hf_hub_download
            
            os.makedirs(self.cache_dir, exist_ok=True)
            
            local_file = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=hf_filename,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✓ Downloaded to: {local_file}")
            return local_file
            
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {file_type} from HuggingFace: {e}")

    def _load_normalization_stats(self, mean_std_path):
        """Load mean and std from pretrain_ds_mean_std.txt"""

        with open(mean_std_path, 'r') as f:
            lines = f.readlines()
            first_line = lines[0].strip()

            if first_line.startswith('{'):
                data = json.loads(first_line)  # Parse ONLY first line
                mean = tuple(data['mean'])
                std = tuple(data['std'])
            else:
                # Plain text format
                mean = tuple(map(float, lines[0].strip().split()))
                std = tuple(map(float, lines[1].strip().split()))

        return mean, std
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict(self, image_path, return_all_probs=False):
        """
        Predict the class of an image
        
        Args:
            image_path: Path to image or PIL Image
            return_all_probs: If True, return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)[0]  # Get probabilities
            
        # Get predicted class
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
        
        # Build result
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels[predicted_class],
            'confidence': confidence,
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                self.class_labels[i]: probs[i].item() 
                for i in range(4)
            }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_all_probs=True)
            results.append(result)
        return results


def print_prediction_result(result):
    """Pretty print prediction results"""
    print("\n" + "="*60)
    print("FSFM-3C UNIFIED DETECTOR - PREDICTION RESULT")
    print("="*60)
    print(f"\n🎯 PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")
    
    if 'all_probabilities' in result:
        print("\n📈 ALL CLASS PROBABILITIES:")
        for label, prob in result['all_probabilities'].items():
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {label:30s} {bar} {prob*100:5.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='FSFM-3C Unified Detector Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='../models/fsfm/checkpoint-min_train_loss.pth',
                        help='Path to checkpoint-min_train_loss.pth')
    parser.add_argument('--mean_std', type=str, default='../models/fsfm/pretrain_ds_mean_std.txt',
                        help='Path to pretrain_ds_mean_std.txt')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("\n🚀 Initializing FSFM-3C Unified Detector...")
    detector = FSFM_UnifiedDetector(
        checkpoint_path=args.checkpoint,
        mean_std_path=args.mean_std,
        device=args.device
    )
    
    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    result = detector.predict(args.image, return_all_probs=True)
    
    # Print results
    print_prediction_result(result)


if __name__ == "__main__":
    # Example usage if run without arguments
    import sys
    
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("FSFM-3C UNIFIED DETECTOR - STANDALONE INFERENCE SCRIPT")
        print("="*70)
        print("\nUsage:")
        print("  python fsfm_unified_detector.py \\")
        print("    --image <path_to_image> \\")
        print("    --checkpoint <path_to_checkpoint-min_train_loss.pth> \\")
        print("    --mean_std <path_to_pretrain_ds_mean_std.txt> \\")
        print("    --device cuda")
        print("\nExample:")
        print("  python fsfm_unified_detector.py \\")
        print("    --image test.jpg \\")
        print("    --checkpoint ./models/fsfm-3c/checkpoint-min_train_loss.pth \\")
        print("    --mean_std ./models/fsfm-3c/pretrain_ds_mean_std.txt \\")
        print("    --device cuda")
        print("\n" + "="*70)
        print("\nDetects 4 classes:")
        print("  • Real/Bonafide faces")
        print("  • Deepfake (face manipulation)")
        print("  • Diffusion/AIGC (AI-generated)")
        print("  • Spoofing/Presentation attacks")
        print("="*70 + "\n")
    else:
        main()
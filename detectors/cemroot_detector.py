#!/usr/bin/env python3
"""
CemRoot Deepfake Detector - Standalone Inference Script
Model: CemRoot/deepfake-detection-aics2025
Architecture: EfficientNetB7 + Custom Attention Mechanism

Conference: 33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)
Author: Emin Cem Koyluoglu
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, regularizers
import argparse
import os

# Disable GPU warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CemRootDetector:
    """
    EfficientNetB7 + Attention mechanism for deepfake detection
    
    Detects 2 classes:
    - Class 0: Fake (AI-Generated)
    - Class 1: Real (Authentic)
    """
    
    # HuggingFace repo for auto-download
    HF_REPO_ID = "CemRoot/deepfake-detection-model"
    HF_FILENAME = "best_model_effatt.h5"
    
    def __init__(self, model_path=None, image_size=128, cache_dir="./models/cemroot"):
        """
        Initialize the detector
        
        Args:
            model_path: Path to best_model_effatt.h5 OR HuggingFace repo ID
                        If None, downloads from HuggingFace automatically
            image_size: Input image size (default: 128)
            cache_dir: Directory to cache downloaded models
        """
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        # Resolve model path (download from HuggingFace if needed)
        self.model_path = self._resolve_model_path(model_path)
        
        print(f"Loading model from: {self.model_path}")
        
        # Try loading the model
        self.model = self._load_model()
        
        print("✓ Model loaded successfully!")
        
        # Class labels
        self.class_labels = {
            0: "FAKE (AI-Generated)",
            1: "REAL (Authentic)"
        }
    
    def _resolve_model_path(self, model_path):
        """
        Resolve model path - download from HuggingFace if needed
        
        Args:
            model_path: Local path, HuggingFace repo ID, or None
            
        Returns:
            Local path to the model file
        """
        # If no path provided, download from HuggingFace
        if model_path is None:
            return self._download_from_huggingface(self.HF_REPO_ID)
        
        # Check if it looks like a local path (starts with . or / or \ or contains .h5)
        is_local_path = (
            model_path.startswith('./') or 
            model_path.startswith('/') or 
            model_path.startswith('\\') or
            model_path.endswith('.h5') or
            os.path.isabs(model_path)
        )
        
        if is_local_path:
            # It's a local path
            if os.path.exists(model_path):
                return model_path
            else:
                # Local file doesn't exist - download from HuggingFace
                print(f"⚠️  Model not found at: {model_path}")
                print(f"📥 Downloading from HuggingFace: {self.HF_REPO_ID}")
                return self._download_from_huggingface(self.HF_REPO_ID)
        else:
            # Assume it's a HuggingFace repo ID (format: namespace/repo_name)
            return self._download_from_huggingface(model_path)
    
    def _download_from_huggingface(self, repo_id):
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "CemRoot/deepfake-detection-model")
            
        Returns:
            Local path to downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"📥 Downloading {self.HF_FILENAME} from {repo_id}...")
            print(f"   (This is ~780MB, may take a few minutes)")
            
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Download the file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=self.HF_FILENAME,
                cache_dir=self.cache_dir,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✓ Downloaded to: {local_path}")
            return local_path
            
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub\n"
                "Or manually download the model from: "
                f"https://huggingface.co/{repo_id}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download from HuggingFace: {e}")
    
    def _attention_block(self, features, depth):
        """
        Custom attention mechanism for enhanced feature extraction
        
        Args:
            features: Input feature tensor
            depth: Feature depth
            
        Returns:
            Global average pooled features with attention
        """
        # Attention pathway
        attn = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(
            layers.Dropout(0.5)(features)
        )
        attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
        attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
        attn = layers.Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(attn)
        
        # Upsampling for attention
        up = layers.Conv2D(depth, (1, 1), padding='same', activation='linear', use_bias=False)
        up_w = np.ones((1, 1, 1, depth), dtype=np.float32)
        up.build((None, None, None, 1))
        up.set_weights([up_w])
        up.trainable = True
        
        attn = up(attn)
        
        # Apply attention
        masked = layers.Multiply()([attn, features])
        
        # Global average pooling with rescaling
        gap_feat = layers.GlobalAveragePooling2D()(masked)
        gap_mask = layers.GlobalAveragePooling2D()(attn)
        gap = layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_feat, gap_mask])
        
        return gap
    
    def _build_model_architecture(self):
        """Build EfficientNetB7 with Attention mechanism"""
        print("Building model architecture from scratch...")
        
        # Base model
        base_model = EfficientNetB7(
            include_top=False,
            weights=None,
            input_shape=(self.image_size, self.image_size, 3)
        )
        base_model.trainable = False
        
        # Feature extraction
        features = base_model.output
        bn_features = layers.BatchNormalization()(features)
        pt_depth = base_model.output_shape[-1]
        
        # Attention block
        gap = self._attention_block(bn_features, pt_depth)
        
        # Classification head
        x = layers.Dropout(0.5)(gap)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(2, activation='softmax')(x)
        
        model = keras.Model(inputs=base_model.input, outputs=outputs)
        
        return model
    
    def _load_model(self):
        """Load the deepfake detection model"""
        try:
            # Custom objects for loading
            def RescaleGAP(tensors):
                return tensors[0] / tensors[1]
            
            # Try loading full model
            try:
                model = keras.models.load_model(
                    self.model_path,
                    custom_objects={
                        'RescaleGAP': RescaleGAP,
                        'attention_block': self._attention_block
                    },
                    compile=False
                )
                print("✓ Loaded full model")
                return model
            except Exception as e:
                print(f"⚠️  Full model load failed: {e}")
                print("🔄 Rebuilding architecture and loading weights...")
                
                # Rebuild and load weights
                model = self._build_model_architecture()
                model.load_weights(self.model_path)
                print("✓ Loaded weights into rebuilt architecture")
                return model
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(self, image_path, method="training_match"):
        """
        Preprocess image for model inference
        
        Args:
            image_path: Path to image or PIL Image
            method: Preprocessing method
                - "training_match": BGR format, 0-255 range (RECOMMENDED - 95% accuracy)
                - "simple_norm": RGB format, 0-1 range (58% accuracy)
                - "efficientnet": EfficientNet ImageNet preprocessing (72% accuracy)
        
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        elif isinstance(image_path, Image.Image):
            image = np.array(image_path.convert('RGB'))
        else:
            image = image_path
        
        # Resize to model input size
        img_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        if method == "training_match":
            # Training Match: RGB -> BGR, keep 0-255 range (BEST PERFORMANCE)
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            arr = img_bgr.astype(np.float32)
            
        elif method == "simple_norm":
            # Simple [0,1] Normalization: RGB, normalize to 0-1
            arr = img_resized.astype(np.float32) / 255.0
            
        elif method == "efficientnet":
            # EfficientNet preprocessing
            from tensorflow.keras.applications.efficientnet import preprocess_input
            arr = preprocess_input(img_resized)
        else:
            # Default to training match
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            arr = img_bgr.astype(np.float32)
        
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
        
        return arr
    
    def predict(self, image_path, method="training_match", return_all_probs=False):
        """
        Predict if image is real or fake
        
        Args:
            image_path: Path to image or PIL Image
            method: Preprocessing method (default: "training_match" for best accuracy)
            return_all_probs: If True, return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path, method=method)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Extract probabilities
        fake_prob = float(predictions[0][0])
        real_prob = float(predictions[0][1])
        
        # Determine predicted class
        predicted_class = 1 if real_prob > fake_prob else 0
        confidence = max(fake_prob, real_prob)
        
        # Build result
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels[predicted_class],
            'confidence': confidence,
            'preprocessing_method': method
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                'FAKE (AI-Generated)': fake_prob,
                'REAL (Authentic)': real_prob
            }
        
        return result
    
    def predict_batch(self, image_paths, method="training_match"):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            method: Preprocessing method
            
        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, method=method, return_all_probs=True)
            results.append(result)
        return results


def print_prediction_result(result):
    """Pretty print prediction results"""
    print("\n" + "="*70)
    print("CEMROOT DEEPFAKE DETECTOR - PREDICTION RESULT")
    print("="*70)
    
    # Color coding
    if result['predicted_class'] == 0:
        emoji = "🚨"
        color = "FAKE"
    else:
        emoji = "✅"
        color = "REAL"
    
    print(f"\n{emoji} PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")
    print(f"🔧 PREPROCESSING: {result['preprocessing_method']}")
    
    if 'all_probabilities' in result:
        print("\n📈 CLASS PROBABILITIES:")
        for label, prob in result['all_probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {label:25s} {bar} {prob*100:5.2f}%")
    
    print("\n" + "="*70)
    print("💡 PREPROCESSING METHODS:")
    print("  • training_match (RECOMMENDED): ~95% accuracy - BGR, 0-255 range")
    print("  • simple_norm: ~58% accuracy - RGB, 0-1 range")
    print("  • efficientnet: ~72% accuracy - ImageNet preprocessing")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='CemRoot Deepfake Detector Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='../models/cemroot/best_model_effatt.h5',
                        help='Path to best_model_effatt.h5')
    parser.add_argument('--method', type=str, default='training_match',
                        choices=['training_match', 'simple_norm', 'efficientnet'],
                        help='Preprocessing method (default: training_match for best accuracy)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size (default: 128)')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("\n🚀 Initializing CemRoot Deepfake Detector...")
    detector = CemRootDetector(
        model_path=args.model,
        image_size=args.image_size
    )
    
    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    print(f"🔧 Using preprocessing: {args.method}")
    
    result = detector.predict(
        args.image,
        method=args.method,
        return_all_probs=True
    )
    
    # Print results
    print_prediction_result(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("CEMROOT DEEPFAKE DETECTOR - STANDALONE INFERENCE SCRIPT")
        print("="*70)
        print("\nArchitecture: EfficientNetB7 + Custom Attention Mechanism")
        print("Conference: AICS 2025")
        print("Author: Emin Cem Koyluoglu")
        print("\nUsage:")
        print("  python cemroot_detector.py \\")
        print("    --image <path_to_image> \\")
        print("    --model <path_to_best_model_effatt.h5> \\")
        print("    --method training_match")
        print("\nExample:")
        print("  python cemroot_detector.py \\")
        print("    --image test.jpg \\")
        print("    --model ./models/cemroot/best_model_effatt.h5 \\")
        print("    --method training_match")
        print("\n" + "="*70)
        print("\nPreprocessing Methods:")
        print("  • training_match (RECOMMENDED): ~95% accuracy")
        print("    - BGR color format, 0-255 range")
        print("  • simple_norm: ~58% accuracy")
        print("    - RGB format, 0-1 range")
        print("  • efficientnet: ~72% accuracy")
        print("    - ImageNet preprocessing")
        print("\n" + "="*70)
        print("\nDetects 2 classes:")
        print("  • FAKE (AI-Generated)")
        print("  • REAL (Authentic)")
        print("="*70 + "\n")
    else:
        main()
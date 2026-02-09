#!/usr/bin/env python3
"""
Bayesian Weight Optimization for Ensemble Deepfake Detection

Finds optimal weights (w1, w2, w3) for combining model predictions.
Implements:
1. Grid Search - Exhaustive search over weight combinations
2. Bayesian Optimization - Smart search using Optuna

The goal: maximize ensemble accuracy on a validation set by
giving more weight to models that perform better on specific data.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import time

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of weight optimization"""
    weights: Dict[str, float]
    accuracy: float
    method: str
    n_trials: int
    timing_seconds: float
    per_model_accuracy: Dict[str, float]


class WeightOptimizer:
    """
    Optimizer for finding optimal ensemble weights
    
    Given predictions from multiple models and ground truth labels,
    finds weights that maximize ensemble accuracy.
    """
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize optimizer
        
        Args:
            model_names: Names of models in ensemble
        """
        self.model_names = model_names or ['fsfm', 'cemroot', 'vit']
        self.weights_file = "optimal_weights.json"
    
    def grid_search(self, 
                    predictions: Dict[str, List[float]],
                    labels: List[int],
                    step: float = 0.1) -> OptimizationResult:
        """
        Exhaustive grid search over weight combinations
        
        Args:
            predictions: Dict mapping model name to list of fake probabilities
            labels: Ground truth labels (1 = fake, 0 = real)
            step: Step size for weight grid (smaller = more precise but slower)
            
        Returns:
            OptimizationResult with best weights found
        """
        start_time = time.time()
        
        # Generate all weight combinations that sum to 1
        weights_list = np.arange(0, 1 + step, step)
        
        best_accuracy = 0
        best_weights = None
        n_trials = 0
        
        # Triple nested loop for 3 models
        for w1 in weights_list:
            for w2 in weights_list:
                w3 = 1 - w1 - w2
                
                # Skip invalid combinations
                if w3 < 0 or w3 > 1:
                    continue
                
                n_trials += 1
                weights = {
                    self.model_names[0]: w1,
                    self.model_names[1]: w2,
                    self.model_names[2]: w3
                }
                
                accuracy = self._evaluate_weights(predictions, labels, weights)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights.copy()
        
        elapsed = time.time() - start_time
        
        # Calculate per-model accuracy
        per_model_acc = self._per_model_accuracy(predictions, labels)
        
        return OptimizationResult(
            weights=best_weights,
            accuracy=best_accuracy,
            method="grid_search",
            n_trials=n_trials,
            timing_seconds=elapsed,
            per_model_accuracy=per_model_acc
        )
    
    def bayesian_optimize(self,
                          predictions: Dict[str, List[float]],
                          labels: List[int],
                          n_trials: int = 100,
                          timeout: Optional[int] = None) -> OptimizationResult:
        """
        Bayesian optimization using Optuna
        
        More efficient than grid search - learns from previous trials
        to focus on promising regions of the weight space.
        
        Args:
            predictions: Dict mapping model name to list of fake probabilities
            labels: Ground truth labels (1 = fake, 0 = real)
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            
        Returns:
            OptimizationResult with best weights found
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not installed. Falling back to grid search.")
            print("   Install with: pip install optuna")
            return self.grid_search(predictions, labels)
        
        start_time = time.time()
        
        def objective(trial):
            # Sample weights
            w1 = trial.suggest_float(self.model_names[0], 0.0, 1.0)
            w2 = trial.suggest_float(self.model_names[1], 0.0, 1.0 - w1)
            w3 = 1.0 - w1 - w2
            
            weights = {
                self.model_names[0]: w1,
                self.model_names[1]: w2,
                self.model_names[2]: w3
            }
            
            return self._evaluate_weights(predictions, labels, weights)
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        elapsed = time.time() - start_time
        
        # Extract best weights
        best_params = study.best_params
        w1 = best_params[self.model_names[0]]
        w2 = best_params[self.model_names[1]]
        w3 = 1.0 - w1 - w2
        
        best_weights = {
            self.model_names[0]: w1,
            self.model_names[1]: w2,
            self.model_names[2]: w3
        }
        
        per_model_acc = self._per_model_accuracy(predictions, labels)
        
        return OptimizationResult(
            weights=best_weights,
            accuracy=study.best_value,
            method="bayesian",
            n_trials=len(study.trials),
            timing_seconds=elapsed,
            per_model_accuracy=per_model_acc
        )
    
    def _evaluate_weights(self,
                          predictions: Dict[str, List[float]],
                          labels: List[int],
                          weights: Dict[str, float]) -> float:
        """
        Evaluate accuracy with given weights
        
        Args:
            predictions: Model predictions
            labels: Ground truth
            weights: Weight for each model
            
        Returns:
            Accuracy (0-1)
        """
        n_samples = len(labels)
        correct = 0
        
        for i in range(n_samples):
            # Weighted average of predictions
            weighted_score = sum(
                weights[name] * predictions[name][i]
                for name in self.model_names
            )
            
            # Threshold at 0.5
            predicted = 1 if weighted_score >= 0.5 else 0
            
            if predicted == labels[i]:
                correct += 1
        
        return correct / n_samples
    
    def _per_model_accuracy(self,
                            predictions: Dict[str, List[float]],
                            labels: List[int]) -> Dict[str, float]:
        """Calculate accuracy for each model individually"""
        results = {}
        n_samples = len(labels)
        
        for name in self.model_names:
            correct = sum(
                1 for i in range(n_samples)
                if (predictions[name][i] >= 0.5) == (labels[i] == 1)
            )
            results[name] = correct / n_samples
        
        return results
    
    def save_weights(self, weights: Dict[str, float], 
                     filepath: str = None) -> str:
        """
        Save optimized weights to JSON file
        
        Args:
            weights: Weight dictionary
            filepath: Output path (default: optimal_weights.json)
            
        Returns:
            Path to saved file
        """
        filepath = filepath or self.weights_file
        
        data = {
            'weights': weights,
            'model_names': self.model_names,
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_weights(self, filepath: str = None) -> Dict[str, float]:
        """
        Load weights from JSON file
        
        Args:
            filepath: Path to weights file
            
        Returns:
            Weight dictionary
        """
        filepath = filepath or self.weights_file
        
        if not os.path.exists(filepath):
            # Return equal weights as default
            default = {name: 1.0 / len(self.model_names) 
                      for name in self.model_names}
            return default
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data['weights']


class ValidationDataLoader:
    """
    Load validation data for weight optimization
    
    Expects directory structure:
    dataset/
    ├── real/
    │   ├── image1.jpg
    │   └── ...
    └── fake/
        ├── image1.jpg
        └── ...
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    def __init__(self, dataset_path: str):
        """
        Initialize loader
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.real_dir = self.dataset_path / "real"
        self.fake_dir = self.dataset_path / "fake"
    
    def load(self) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels
        
        Returns:
            Tuple of (image_paths, labels)
            Labels: 0 = real, 1 = fake
        """
        image_paths = []
        labels = []
        
        # Load real images
        if self.real_dir.exists():
            for img_path in self.real_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img_path))
                    labels.append(0)  # Real
        
        # Load fake images
        if self.fake_dir.exists():
            for img_path in self.fake_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img_path))
                    labels.append(1)  # Fake
        
        return image_paths, labels
    
    def summary(self) -> Dict:
        """Get dataset summary"""
        paths, labels = self.load()
        
        return {
            'total_images': len(paths),
            'real_count': labels.count(0),
            'fake_count': labels.count(1),
            'real_dir': str(self.real_dir),
            'fake_dir': str(self.fake_dir)
        }


def run_optimization(ensemble_detector,
                     dataset_path: str,
                     method: str = 'bayesian',
                     n_trials: int = 100,
                     max_samples: int = None,
                     output_path: str = 'optimal_weights.json') -> OptimizationResult:
    """
    Convenience function to run full optimization pipeline
    
    Args:
        ensemble_detector: Initialized EnsembleDeepfakeDetector
        dataset_path: Path to validation dataset
        method: 'grid' or 'bayesian'
        n_trials: Number of trials for bayesian optimization
        max_samples: Maximum number of images to process (None = all)
        output_path: Where to save weights
        
    Returns:
        OptimizationResult
    """
    print("="*70)
    print("🔧 ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    
    # Load dataset
    loader = ValidationDataLoader(dataset_path)
    summary = loader.summary()
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"   Real images: {summary['real_count']}")
    print(f"   Fake images: {summary['fake_count']}")
    print(f"   Total: {summary['total_images']}")
    
    if summary['total_images'] == 0:
        raise ValueError("No images found in dataset")
    
    image_paths, labels = loader.load()
    
    # Sample if max_samples specified
    if max_samples and max_samples < len(image_paths):
        import random
        print(f"\n🎲 Sampling {max_samples} images (stratified)...")
        
        # Stratified sampling to maintain class balance
        real_indices = [i for i, l in enumerate(labels) if l == 0]
        fake_indices = [i for i, l in enumerate(labels) if l == 1]
        
        # Calculate samples per class
        n_real = min(len(real_indices), max_samples // 2)
        n_fake = min(len(fake_indices), max_samples - n_real)
        
        # Random sample from each class
        random.shuffle(real_indices)
        random.shuffle(fake_indices)
        sampled_indices = real_indices[:n_real] + fake_indices[:n_fake]
        random.shuffle(sampled_indices)  # Mix them up
        
        image_paths = [image_paths[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        
        print(f"   Sampled: {n_real} real + {n_fake} fake = {len(image_paths)} total")
    
    # Get predictions from each model
    print(f"\n🔍 Getting predictions from all models...")
    
    predictions = {
        'fsfm': [],
        'cemroot': [],
        'vit': []
    }
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i + 1}/{len(image_paths)}...")
        
        result = ensemble_detector.predict(img_path)
        
        # Extract fake probability from each model
        for model_name in predictions.keys():
            model_data = result['models'][model_name]
            # Use confidence if fake, 1-confidence if real
            if model_data['is_fake']:
                fake_prob = model_data['confidence']
            else:
                fake_prob = 1 - model_data['confidence']
            predictions[model_name].append(fake_prob)
    
    print("   ✓ All predictions collected")
    
    # Run optimization
    optimizer = WeightOptimizer()
    
    if method == 'grid':
        print(f"\n⏳ Running grid search...")
        result = optimizer.grid_search(predictions, labels, step=0.05)
    else:
        print(f"\n⏳ Running Bayesian optimization ({n_trials} trials)...")
        result = optimizer.bayesian_optimize(predictions, labels, n_trials=n_trials)
    
    # Print results
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n🎯 Best Weights:")
    for name, weight in result.weights.items():
        print(f"   {name.upper()}: {weight:.3f}")
    
    print(f"\n📈 Ensemble Accuracy: {result.accuracy*100:.2f}%")
    
    print(f"\n📊 Individual Model Accuracy:")
    for name, acc in result.per_model_accuracy.items():
        print(f"   {name.upper()}: {acc*100:.2f}%")
    
    print(f"\n⏱️  Optimization Time: {result.timing_seconds:.2f}s")
    print(f"   Trials Evaluated: {result.n_trials}")
    print(f"   Method: {result.method}")
    
    # Save weights
    saved_path = optimizer.save_weights(result.weights, output_path)
    print(f"\n💾 Weights saved to: {saved_path}")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize ensemble weights for deepfake detection'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to validation dataset (with real/ and fake/ subdirs)')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['grid', 'bayesian'],
                        help='Optimization method')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials for Bayesian optimization')
    parser.add_argument('--output', type=str, default='optimal_weights.json',
                        help='Output path for weights')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("To run optimization, you need an initialized ensemble detector.")
    print("Example usage in Python:")
    print("="*70)
    print("""
from detectors.ensemble_detector import EnsembleDeepfakeDetector
from detectors.weight_optimizer import run_optimization

# Initialize your ensemble
ensemble = EnsembleDeepfakeDetector(
    fsfm_config={'checkpoint': '...', 'mean_std': '...'},
    cemroot_config={'model_path': '...'},
    vit_config={'model_name': '...'}
)

# Run optimization
result = run_optimization(
    ensemble_detector=ensemble,
    dataset_path='./data/validation',
    method='bayesian',
    n_trials=100,
    output_path='optimal_weights.json'
)

print(f"Best weights: {result.weights}")
print(f"Accuracy: {result.accuracy}")
""")

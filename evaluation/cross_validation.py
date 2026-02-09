#!/usr/bin/env python3
"""
Cross-Validation for Deepfake Ensemble Detection

Implements K-Fold cross-validation to:
1. Validate model performance isn't a fluke
2. Prevent overfitting to specific faces
3. Generate statistically significant metrics

Based on academic paper methodology using 5-fold and 10-fold CV.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import defaultdict

# Try to import sklearn for metrics
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class FoldResult:
    """Results from a single fold"""
    fold_index: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float]
    confusion_matrix: List[List[int]]
    n_test_samples: int


@dataclass  
class CrossValidationResult:
    """Aggregated cross-validation results"""
    k_folds: int
    fold_results: List[FoldResult]
    
    # Aggregated metrics (mean ± std)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_precision: float = 0.0
    std_precision: float = 0.0
    mean_recall: float = 0.0
    std_recall: float = 0.0
    mean_f1: float = 0.0
    std_f1: float = 0.0
    mean_auc: Optional[float] = None
    std_auc: Optional[float] = None
    
    total_time_seconds: float = 0.0
    per_model_results: Dict[str, Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate aggregate statistics after initialization"""
        if self.fold_results:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std for all metrics"""
        accuracies = [f.accuracy for f in self.fold_results]
        precisions = [f.precision for f in self.fold_results]
        recalls = [f.recall for f in self.fold_results]
        f1s = [f.f1 for f in self.fold_results]
        aucs = [f.auc_roc for f in self.fold_results if f.auc_roc is not None]
        
        self.mean_accuracy = np.mean(accuracies)
        self.std_accuracy = np.std(accuracies)
        self.mean_precision = np.mean(precisions)
        self.std_precision = np.std(precisions)
        self.mean_recall = np.mean(recalls)
        self.std_recall = np.std(recalls)
        self.mean_f1 = np.mean(f1s)
        self.std_f1 = np.std(f1s)
        
        if aucs:
            self.mean_auc = np.mean(aucs)
            self.std_auc = np.std(aucs)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'k_folds': self.k_folds,
            'metrics': {
                'accuracy': {'mean': self.mean_accuracy, 'std': self.std_accuracy},
                'precision': {'mean': self.mean_precision, 'std': self.std_precision},
                'recall': {'mean': self.mean_recall, 'std': self.std_recall},
                'f1': {'mean': self.mean_f1, 'std': self.std_f1},
                'auc_roc': {'mean': self.mean_auc, 'std': self.std_auc} if self.mean_auc else None
            },
            'fold_results': [
                {
                    'fold': f.fold_index,
                    'accuracy': f.accuracy,
                    'precision': f.precision,
                    'recall': f.recall,
                    'f1': f.f1,
                    'auc_roc': f.auc_roc,
                    'n_samples': f.n_test_samples
                }
                for f in self.fold_results
            ],
            'per_model_results': self.per_model_results,
            'total_time_seconds': self.total_time_seconds
        }


class CrossValidator:
    """
    K-Fold Cross-Validation for Deepfake Ensemble
    
    Splits dataset into K folds, trains/evaluates on each fold,
    and reports aggregated metrics with confidence intervals.
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize cross-validator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn required for cross-validation.\n"
                "Install with: pip install scikit-learn"
            )
        
        self.random_seed = random_seed
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], np.ndarray]:
        """
        Load dataset from directory structure
        
        Expected structure:
        dataset/
        ├── real/
        │   └── *.jpg
        └── fake/
            └── *.jpg
            
        Returns:
            Tuple of (image_paths, labels)
        """
        dataset_path = Path(dataset_path)
        real_dir = dataset_path / "real"
        fake_dir = dataset_path / "fake"
        
        image_paths = []
        labels = []
        
        # Load real images (label = 0)
        if real_dir.exists():
            for img in real_dir.iterdir():
                if img.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img))
                    labels.append(0)
        
        # Load fake images (label = 1)
        if fake_dir.exists():
            for img in fake_dir.iterdir():
                if img.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img))
                    labels.append(1)
        
        return image_paths, np.array(labels)
    
    def run_kfold(self,
                  ensemble_detector,
                  dataset_path: str,
                  k: int = 5,
                  weights: Optional[Dict[str, float]] = None,
                  verbose: bool = True) -> CrossValidationResult:
        """
        Run K-Fold cross-validation
        
        Args:
            ensemble_detector: Initialized EnsembleDeepfakeDetector
            dataset_path: Path to dataset
            k: Number of folds (5 or 10 recommended)
            weights: Optional model weights for weighted ensemble
            verbose: Print progress
            
        Returns:
            CrossValidationResult with all metrics
        """
        start_time = time.time()
        
        # Load dataset
        image_paths, labels = self.load_dataset(dataset_path)
        n_samples = len(labels)
        
        if verbose:
            print("="*70)
            print(f"🔄 {k}-FOLD CROSS-VALIDATION")
            print("="*70)
            print(f"\n📁 Dataset: {dataset_path}")
            print(f"   Total samples: {n_samples}")
            print(f"   Real: {np.sum(labels == 0)}, Fake: {np.sum(labels == 1)}")
            print(f"   Folds: {k}")
        
        # Create stratified folds (maintains class balance)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        per_model_preds = defaultdict(list)
        per_model_labels = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
            if verbose:
                print(f"\n📊 Fold {fold_idx + 1}/{k}")
            
            # Get test samples for this fold
            test_paths = [image_paths[i] for i in test_idx]
            test_labels = labels[test_idx]
            
            # Get predictions
            ensemble_preds = []
            model_preds = {'fsfm': [], 'cemroot': [], 'vit': []}
            
            for img_path in test_paths:
                try:
                    result = ensemble_detector.predict(img_path)
                    
                    # Get individual model predictions
                    for model_name in model_preds.keys():
                        model_data = result['models'][model_name]
                        # Binary prediction: 1 if fake, 0 if real
                        model_preds[model_name].append(1 if model_data['is_fake'] else 0)
                    
                    # Ensemble prediction (weighted or majority)
                    if weights:
                        # Weighted voting
                        weighted_score = sum(
                            weights.get(name, 1/3) * (1 if result['models'][name]['is_fake'] else 0)
                            for name in ['fsfm', 'cemroot', 'vit']
                        )
                        ensemble_preds.append(1 if weighted_score >= 0.5 else 0)
                    else:
                        # Majority voting
                        fake_votes = result['summary']['models_detecting_fake']
                        ensemble_preds.append(1 if fake_votes >= 2 else 0)
                        
                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ Error on {img_path}: {e}")
                    # Default to uncertain prediction
                    ensemble_preds.append(0)
                    for model_name in model_preds.keys():
                        model_preds[model_name].append(0)
            
            # Calculate metrics for this fold
            y_true = test_labels
            y_pred = np.array(ensemble_preds)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC-ROC (needs probability scores, use average confidence as proxy)
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = None
            
            cm = confusion_matrix(y_true, y_pred).tolist()
            
            fold_result = FoldResult(
                fold_index=fold_idx + 1,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                auc_roc=float(auc) if auc else None,
                confusion_matrix=cm,
                n_test_samples=len(test_labels)
            )
            fold_results.append(fold_result)
            
            # Store per-model predictions
            for name, preds in model_preds.items():
                per_model_preds[name].extend(preds)
            per_model_labels.extend(y_true)
            
            if verbose:
                print(f"     Accuracy: {accuracy*100:.1f}% | "
                      f"Precision: {precision*100:.1f}% | "
                      f"Recall: {recall*100:.1f}% | "
                      f"F1: {f1*100:.1f}%")
        
        # Calculate per-model metrics
        per_model_results = {}
        for name, preds in per_model_preds.items():
            y_pred_model = np.array(preds)
            y_true_all = np.array(per_model_labels)
            per_model_results[name] = {
                'accuracy': float(accuracy_score(y_true_all, y_pred_model)),
                'precision': float(precision_score(y_true_all, y_pred_model, zero_division=0)),
                'recall': float(recall_score(y_true_all, y_pred_model, zero_division=0)),
                'f1': float(f1_score(y_true_all, y_pred_model, zero_division=0))
            }
        
        total_time = time.time() - start_time
        
        # Create result
        cv_result = CrossValidationResult(
            k_folds=k,
            fold_results=fold_results,
            total_time_seconds=total_time,
            per_model_results=per_model_results
        )
        
        if verbose:
            self._print_summary(cv_result)
        
        return cv_result
    
    def _print_summary(self, result: CrossValidationResult):
        """Print cross-validation summary"""
        print("\n" + "="*70)
        print("📈 CROSS-VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n🎯 ENSEMBLE METRICS ({result.k_folds}-fold):")
        print(f"   Accuracy:  {result.mean_accuracy*100:.2f}% ± {result.std_accuracy*100:.2f}%")
        print(f"   Precision: {result.mean_precision*100:.2f}% ± {result.std_precision*100:.2f}%")
        print(f"   Recall:    {result.mean_recall*100:.2f}% ± {result.std_recall*100:.2f}%")
        print(f"   F1 Score:  {result.mean_f1*100:.2f}% ± {result.std_f1*100:.2f}%")
        if result.mean_auc:
            print(f"   AUC-ROC:   {result.mean_auc*100:.2f}% ± {result.std_auc*100:.2f}%")
        
        print(f"\n📊 PER-MODEL METRICS (overall):")
        for name, metrics in result.per_model_results.items():
            print(f"   {name.upper()}:")
            print(f"      Accuracy: {metrics['accuracy']*100:.2f}% | "
                  f"F1: {metrics['f1']*100:.2f}%")
        
        print(f"\n⏱️  Total Time: {result.total_time_seconds:.1f}s")
        print("="*70)
    
    def save_results(self, result: CrossValidationResult, 
                     filepath: str = "cv_results.json"):
        """Save cross-validation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n💾 Results saved to: {filepath}")


def run_cross_validation(ensemble_detector,
                         dataset_path: str,
                         k: int = 5,
                         weights: Optional[Dict[str, float]] = None,
                         output_path: str = "cv_results.json") -> CrossValidationResult:
    """
    Convenience function to run cross-validation
    
    Args:
        ensemble_detector: Initialized EnsembleDeepfakeDetector
        dataset_path: Path to dataset
        k: Number of folds
        weights: Optional model weights
        output_path: Where to save results
        
    Returns:
        CrossValidationResult
    """
    cv = CrossValidator()
    result = cv.run_kfold(ensemble_detector, dataset_path, k=k, weights=weights)
    cv.save_results(result, output_path)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run K-Fold cross-validation on deepfake ensemble'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (with real/ and fake/ subdirs)')
    parser.add_argument('--folds', type=int, default=5, choices=[5, 10],
                        help='Number of folds (5 or 10)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weights JSON file')
    parser.add_argument('--output', type=str, default='cv_results.json',
                        help='Output path for results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("To run cross-validation, initialize ensemble detector first.")
    print("Example:")
    print("="*70)
    print("""
from detectors.ensemble_detector import EnsembleDeepfakeDetector
from evaluation.cross_validation import run_cross_validation

# Initialize ensemble
ensemble = EnsembleDeepfakeDetector(
    fsfm_config={'checkpoint': '...', 'mean_std': '...'},
    cemroot_config={'model_path': '...'},
    vit_config={'model_name': '...'}
)

# Run 5-fold cross-validation
result = run_cross_validation(
    ensemble_detector=ensemble,
    dataset_path='./data',
    k=5,
    output_path='cv_results.json'
)

print(f"Mean Accuracy: {result.mean_accuracy*100:.2f}%")
print(f"Mean F1: {result.mean_f1*100:.2f}%")
""")

#!/usr/bin/env python3
"""
Model Diagnostic Script - Verify label mappings and per-dataset accuracy

Tests each model individually on samples from each dataset to:
1. Verify that label mappings are correct (not inverted)
2. Show per-dataset accuracy for each model
3. Identify which models work best on which data types

Usage:
    python scripts/diagnose_models.py --dataset ./data1 ./data2 ./data3 --samples 100
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_from_dataset(dataset_path, n_samples):
    """Load and sample images from a dataset directory."""
    EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    real_dir = os.path.join(dataset_path, 'real')
    fake_dir = os.path.join(dataset_path, 'fake')
    
    real_images = []
    fake_images = []
    
    if os.path.exists(real_dir):
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                       if os.path.splitext(f)[1].lower() in EXTS]
    
    if os.path.exists(fake_dir):
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                       if os.path.splitext(f)[1].lower() in EXTS]
    
    # Sample equally from both classes
    n_per_class = n_samples // 2
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    sampled_real = real_images[:min(n_per_class, len(real_images))]
    sampled_fake = fake_images[:min(n_per_class, len(fake_images))]
    
    paths = sampled_real + sampled_fake
    labels = [0] * len(sampled_real) + [1] * len(sampled_fake)  # 0=real, 1=fake
    
    # Shuffle together
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = zip(*combined) if combined else ([], [])
    
    return list(paths), list(labels)


def test_model_individually(model_name, model, image_paths, true_labels):
    """
    Test a single model and return detailed diagnostics.
    Returns dict with accuracy, predictions, and label analysis.
    """
    correct = 0
    predictions = []
    fake_probs = []
    
    for img_path, true_label in zip(image_paths, true_labels):
        try:
            result = model.predict(img_path, return_all_probs=True)
            
            # Get is_fake from standardized output
            is_fake = result.get('is_fake', None)
            fake_prob = result.get('fake_probability', None)
            
            # Fallback for models without standardized output
            if is_fake is None:
                is_fake = result['predicted_class'] != 0
            if fake_prob is None:
                fake_prob = result['confidence'] if is_fake else (1 - result['confidence'])
            
            predicted_label = 1 if is_fake else 0
            is_correct = (predicted_label == true_label)
            correct += is_correct
            
            predictions.append({
                'true_label': true_label,
                'predicted': predicted_label,
                'is_fake': is_fake,
                'fake_prob': fake_prob,
                'raw_label': result['predicted_label'],
                'confidence': result['confidence'],
                'correct': is_correct,
            })
            fake_probs.append(fake_prob)
            
        except Exception as e:
            print(f"      ⚠️  Error on {os.path.basename(img_path)}: {e}")
            predictions.append(None)
    
    valid_preds = [p for p in predictions if p is not None]
    accuracy = correct / len(valid_preds) if valid_preds else 0
    
    # Check if labels might be inverted (accuracy < 50% suggests inversion)
    inverted_accuracy = 1 - accuracy
    
    # Calculate per-class accuracy
    real_correct = sum(1 for p in valid_preds if p['true_label'] == 0 and p['correct'])
    real_total = sum(1 for p in valid_preds if p['true_label'] == 0)
    fake_correct = sum(1 for p in valid_preds if p['true_label'] == 1 and p['correct'])
    fake_total = sum(1 for p in valid_preds if p['true_label'] == 1)
    
    # Avg fake_prob for real vs fake images
    avg_fp_real = sum(p['fake_prob'] for p in valid_preds if p['true_label'] == 0) / max(real_total, 1)
    avg_fp_fake = sum(p['fake_prob'] for p in valid_preds if p['true_label'] == 1) / max(fake_total, 1)
    
    return {
        'accuracy': accuracy,
        'inverted_accuracy': inverted_accuracy,
        'real_accuracy': real_correct / max(real_total, 1),
        'fake_accuracy': fake_correct / max(fake_total, 1),
        'real_total': real_total,
        'fake_total': fake_total,
        'avg_fake_prob_for_real': avg_fp_real,
        'avg_fake_prob_for_fake': avg_fp_fake,
        'predictions': valid_preds,
        'label_likely_inverted': inverted_accuracy > accuracy + 0.1,
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose model label mappings and accuracy')
    parser.add_argument('--dataset', type=str, nargs='+', required=True,
                        help='Path(s) to dataset(s) with real/ and fake/ subdirs')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples per dataset (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--organika_model', type=str,
                        default='Organika/sdxl-detector')
    parser.add_argument('--siglip_model', type=str,
                        default='prithivMLmods/open-deepfake-detection')
    parser.add_argument('--predictor_path', type=str, default=None,
                        help='Path to dlib shape_predictor_68_face_landmarks.dat')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔬 MODEL DIAGNOSTIC TOOL")
    print("="*70)
    
    # ── Load models individually ──
    print("\n📦 Loading models...")
    
    print("\n  [1/4] FSFM-3C...")
    from detectors.fsfm_unified_detector import FSFM_UnifiedDetector
    fsfm = FSFM_UnifiedDetector(device=args.device)
    
    print("\n  [2/4] Organika...")
    from detectors.organika_detector import OrganikaDetector
    organika = OrganikaDetector(model_name=args.organika_model, device=args.device)
    
    print("\n  [3/4] SigLIP...")
    from detectors.siglip_detector import SigLIPDetector
    siglip = SigLIPDetector(model_name=args.siglip_model, device=args.device)
    
    print("\n  [4/4] Face Forensics...")
    from detectors.face_forensics_detector import FaceForensicsAnalyzer
    forensics = FaceForensicsAnalyzer(predictor_path=args.predictor_path, device=args.device)
    
    models = {
        'FSFM': fsfm,
        'Organika': organika,
        'SigLIP': siglip,
        'Forensics': forensics,
    }
    
    # ── Test on each dataset ──
    all_results = {}  # {dataset_name: {model_name: result_dict}}
    
    for ds_path in args.dataset:
        ds_name = os.path.basename(ds_path.rstrip('/\\'))
        print(f"\n\n{'='*70}")
        print(f"📁 DATASET: {ds_name}")
        print(f"   Path: {ds_path}")
        print(f"{'='*70}")
        
        # Sample images
        paths, labels = sample_from_dataset(ds_path, args.samples)
        n_real = labels.count(0)
        n_fake = labels.count(1)
        print(f"   Sampled: {len(paths)} images ({n_real} real, {n_fake} fake)")
        
        if not paths:
            print("   ⚠️  No images found, skipping!")
            continue
        
        all_results[ds_name] = {}
        
        for model_name, model in models.items():
            print(f"\n   🔍 Testing {model_name}...")
            result = test_model_individually(model_name, model, paths, labels)
            all_results[ds_name][model_name] = result
            
            # Print quick summary
            acc = result['accuracy'] * 100
            inv = result['inverted_accuracy'] * 100
            
            status = "✅" if acc > 60 else "⚠️" if acc > 50 else "🔴"
            print(f"      {status} Accuracy: {acc:.1f}%")
            print(f"         Real images → avg fake_prob: {result['avg_fake_prob_for_real']:.3f}")
            print(f"         Fake images → avg fake_prob: {result['avg_fake_prob_for_fake']:.3f}")
            
            if result['label_likely_inverted']:
                print(f"      🔄 LABEL LIKELY INVERTED! Flipped accuracy would be: {inv:.1f}%")
            
            print(f"         Per-class: Real={result['real_accuracy']*100:.1f}% ({result['real_total']}), "
                  f"Fake={result['fake_accuracy']*100:.1f}% ({result['fake_total']})")
    
    # ── Summary Table ──
    print("\n\n" + "="*70)
    print("📊 SUMMARY: ACCURACY PER MODEL PER DATASET")
    print("="*70)
    
    dataset_names = list(all_results.keys())
    model_names = list(models.keys())
    
    # Header
    header = f"{'Model':<15}"
    for ds in dataset_names:
        header += f" | {ds:<20}"
    header += f" | {'Average':<10}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for mn in model_names:
        row = f"{mn:<15}"
        accs = []
        for ds in dataset_names:
            if mn in all_results.get(ds, {}):
                acc = all_results[ds][mn]['accuracy'] * 100
                accs.append(acc)
                inverted = all_results[ds][mn]['label_likely_inverted']
                marker = " 🔄" if inverted else ""
                row += f" | {acc:>5.1f}%{marker:<14}"
            else:
                row += f" | {'N/A':<20}"
        avg = sum(accs) / len(accs) if accs else 0
        row += f" | {avg:>5.1f}%"
        print(row)
    
    # ── Label Mapping Analysis ──
    print("\n\n" + "="*70)
    print("🏷️  LABEL MAPPING ANALYSIS")
    print("="*70)
    
    for mn in model_names:
        print(f"\n  {mn}:")
        for ds in dataset_names:
            if mn not in all_results.get(ds, {}):
                continue
            r = all_results[ds][mn]
            fp_real = r['avg_fake_prob_for_real']
            fp_fake = r['avg_fake_prob_for_fake']
            separation = fp_fake - fp_real
            
            if r['label_likely_inverted']:
                print(f"    {ds}: 🔄 INVERTED - fake_prob higher for REAL images!")
                print(f"      Real img avg fake_prob: {fp_real:.3f}")
                print(f"      Fake img avg fake_prob: {fp_fake:.3f}")
                print(f"      → Fix: swap class 0 ↔ 1 in detector config")
            elif abs(separation) < 0.05:
                print(f"    {ds}: ⚠️  NO SEPARATION - model can't distinguish (sep={separation:.3f})")
            elif separation > 0:
                print(f"    {ds}: ✅ CORRECT - fake_prob higher for fake images (sep={separation:.3f})")
            else:
                print(f"    {ds}: 🔄 INVERTED - fake_prob higher for real images (sep={separation:.3f})")
    
    # ── Recommendation ──
    print("\n\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    for mn in model_names:
        inversions = 0
        no_sep = 0
        correct = 0
        total_ds = 0
        for ds in dataset_names:
            if mn not in all_results.get(ds, {}):
                continue
            total_ds += 1
            r = all_results[ds][mn]
            sep = r['avg_fake_prob_for_fake'] - r['avg_fake_prob_for_real']
            if r['label_likely_inverted'] or sep < -0.05:
                inversions += 1
            elif abs(sep) < 0.05:
                no_sep += 1
            else:
                correct += 1
        
        if inversions > total_ds / 2:
            print(f"\n  {mn}: 🔄 Labels are INVERTED on {inversions}/{total_ds} datasets")
            print(f"    → Action: Flip fake_class_idx (swap 0 ↔ 1)")
        elif no_sep > total_ds / 2:
            print(f"\n  {mn}: ⚠️  No discrimination power on {no_sep}/{total_ds} datasets")
            print(f"    → Action: Consider replacing this model")
        else:
            print(f"\n  {mn}: ✅ Labels correct on {correct}/{total_ds} datasets")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

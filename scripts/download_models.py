#!/usr/bin/env python3
"""
Download all models into a local directory (for PVC pre-population).

Run this ONCE to populate the PVC volume, then set HF_LOCAL_ONLY=true
so the API container never needs internet access at runtime.

Usage:
    python scripts/download_models.py --output /models
    python scripts/download_models.py --output ./models  # local dev
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_fsfm(output_dir):
    """Download FSFM-3C checkpoint and mean_std from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    dest = os.path.join(output_dir, "fsfm-3c")
    os.makedirs(dest, exist_ok=True)
    
    print(f"\n📥 [1/4] Downloading FSFM-3C → {dest}")
    
    # Download checkpoint
    ckpt = hf_hub_download(
        repo_id="Wolowolo/fsfm-3c",
        filename="checkpoint-best.pth",
        local_dir=dest,
    )
    print(f"   ✓ Checkpoint: {ckpt}")
    
    # Download mean_std
    ms = hf_hub_download(
        repo_id="Wolowolo/fsfm-3c",
        filename="mean_std.pkl",
        local_dir=dest,
    )
    print(f"   ✓ mean_std: {ms}")


def download_organika(output_dir):
    """Download Organika SDXL detector."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    
    dest = os.path.join(output_dir, "organika-sdxl-detector")
    os.makedirs(dest, exist_ok=True)
    
    print(f"\n📥 [2/4] Downloading Organika → {dest}")
    
    processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
    model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
    
    processor.save_pretrained(dest)
    model.save_pretrained(dest)
    print(f"   ✓ Saved to {dest}")


def download_siglip(output_dir):
    """Download SigLIP deepfake detector."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    
    dest = os.path.join(output_dir, "siglip-deepfake-detection")
    os.makedirs(dest, exist_ok=True)
    
    print(f"\n📥 [3/4] Downloading SigLIP → {dest}")
    
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/open-deepfake-detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/open-deepfake-detection")
    
    processor.save_pretrained(dest)
    model.save_pretrained(dest)
    print(f"   ✓ Saved to {dest}")


def download_dlib(output_dir):
    """Download dlib shape predictor."""
    import urllib.request
    import bz2
    
    dest = os.path.join(output_dir, "dlib")
    os.makedirs(dest, exist_ok=True)
    dat_path = os.path.join(dest, "shape_predictor_68_face_landmarks.dat")
    
    if os.path.exists(dat_path):
        print(f"\n📥 [4/4] dlib shape predictor already exists: {dat_path}")
        return
    
    print(f"\n📥 [4/4] Downloading dlib shape predictor → {dest}")
    
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = dat_path + ".bz2"
    
    print(f"   Downloading (~65MB)...")
    urllib.request.urlretrieve(url, bz2_path)
    
    print(f"   Extracting...")
    with bz2.BZ2File(bz2_path) as fr, open(dat_path, 'wb') as fw:
        fw.write(fr.read())
    
    os.remove(bz2_path)
    print(f"   ✓ Saved to {dat_path}")


def main():
    parser = argparse.ArgumentParser(description='Download all models for PVC')
    parser.add_argument('--output', type=str, default='/models',
                        help='Output directory (default: /models)')
    parser.add_argument('--skip-dlib', action='store_true',
                        help='Skip dlib shape predictor download')
    args = parser.parse_args()
    
    print("=" * 70)
    print("📦 DOWNLOADING ALL MODELS FOR DEEPFAKE ENSEMBLE")
    print(f"   Output: {args.output}")
    print("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    download_fsfm(args.output)
    download_organika(args.output)
    download_siglip(args.output)
    
    if not args.skip_dlib:
        download_dlib(args.output)
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS DOWNLOADED")
    print("=" * 70)
    print(f"\nSet these ENV vars to use local models:")
    print(f"  FSFM_MODEL_PATH={args.output}/fsfm-3c")
    print(f"  ORGANIKA_MODEL={args.output}/organika-sdxl-detector")
    print(f"  SIGLIP_MODEL={args.output}/siglip-deepfake-detection")
    print(f"  PREDICTOR_PATH={args.output}/dlib/shape_predictor_68_face_landmarks.dat")
    print(f"  HF_LOCAL_ONLY=true")


if __name__ == "__main__":
    main()

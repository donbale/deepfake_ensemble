#!/usr/bin/env python3
"""
CLI Script for Preprocessing Analysis

Runs FFT and facial landmark analysis on an image to detect
potential deepfake artifacts.

Usage:
    python scripts/analyze_image.py --image test.jpg
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.preprocessing import full_preprocessing


def main():
    parser = argparse.ArgumentParser(
        description='Analyze image for deepfake artifacts'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save JSON results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    print(f"\n🔍 Analyzing: {args.image}\n")
    
    result = full_preprocessing(args.image)
    
    print("="*70)
    print("PREPROCESSING ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📊 FFT Analysis:")
    print(f"   Anomaly Score: {result['fft_analysis']['anomaly_score']:.3f}")
    print(f"   High Freq Ratio: {result['fft_analysis']['high_freq_ratio']:.4f}")
    print(f"   {result['fft_analysis']['interpretation']}")
    
    print(f"\n👤 Landmark Analysis:")
    print(f"   Consistency Score: {result['landmark_analysis']['consistency_score']:.3f}")
    print(f"   Symmetry Score: {result['landmark_analysis']['symmetry_score']:.3f}")
    if result['landmark_analysis'].get('issues'):
        print(f"   Issues: {', '.join(result['landmark_analysis']['issues'])}")
    print(f"   {result['landmark_analysis']['interpretation']}")
    
    print(f"\n🎯 COMBINED SUSPICION SCORE: {result['combined_suspicion_score']:.3f}")
    print(f"   {result['interpretation']}")
    print(f"\n💡 {result['recommendation']}")
    print("="*70 + "\n")
    
    # Save to file if requested
    if args.output:
        import json
        # Convert numpy arrays to lists for JSON serialization
        output_result = {
            'combined_suspicion_score': result['combined_suspicion_score'],
            'interpretation': result['interpretation'],
            'recommendation': result['recommendation'],
            'fft_analysis': {
                'anomaly_score': result['fft_analysis']['anomaly_score'],
                'high_freq_ratio': result['fft_analysis']['high_freq_ratio'],
                'interpretation': result['fft_analysis']['interpretation']
            },
            'landmark_analysis': {
                'consistency_score': result['landmark_analysis'].get('consistency_score'),
                'symmetry_score': result['landmark_analysis'].get('symmetry_score'),
                'issues': result['landmark_analysis'].get('issues', []),
                'interpretation': result['landmark_analysis'].get('interpretation')
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_result, f, indent=2)
        print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

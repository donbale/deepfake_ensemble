#!/usr/bin/env python3
"""
Advanced Preprocessing for Deepfake Detection

Implements two key analysis techniques:
1. FFT (Frequency Domain) Analysis - Detects unnatural frequency patterns
2. Facial Landmark Consistency - Checks geometric proportions

Many deepfakes are pixel-perfect but show anomalies in:
- High-frequency components (FFT reveals artifacts from upscaling/generation)
- Facial geometry (AI struggles with consistent inter-landmark distances)
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, Optional, Union
import os

# Try to import dlib for landmark detection, fall back to mediapipe
LANDMARK_BACKEND = None
try:
    import dlib
    LANDMARK_BACKEND = "dlib"
except ImportError:
    try:
        import mediapipe as mp
        LANDMARK_BACKEND = "mediapipe"
    except ImportError:
        pass


class FFTAnalyzer:
    """
    Frequency Domain Analysis for Deepfake Detection
    
    Deepfakes often show artifacts in the frequency domain:
    - GAN-generated images have characteristic high-frequency patterns
    - Upscaling artifacts appear as regular patterns in FFT
    - Blending boundaries create frequency discontinuities
    """
    
    def __init__(self, target_size: int = 256):
        """
        Initialize FFT analyzer
        
        Args:
            target_size: Resize images to this size for consistent analysis
        """
        self.target_size = target_size
    
    def analyze(self, image_path: Union[str, Image.Image]) -> Dict:
        """
        Perform FFT analysis on an image
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Dictionary with:
            - magnitude_spectrum: 2D array of frequency magnitudes
            - anomaly_score: 0-1 score (higher = more suspicious)
            - radial_profile: Average magnitude at each frequency
            - high_freq_ratio: Ratio of high to low frequency energy
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Convert PIL Image to cv2 format
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.target_size, self.target_size))
        
        # Apply windowing to reduce edge effects
        window = np.hanning(self.target_size)
        window_2d = np.outer(window, window)
        windowed = gray * window_2d
        
        # Compute 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Log scale for visualization (avoid log(0))
        magnitude_log = np.log1p(magnitude)
        
        # Compute radial average (azimuthal integration)
        radial_profile = self._compute_radial_profile(magnitude)
        
        # Compute frequency band ratios
        high_freq_ratio = self._compute_frequency_ratio(magnitude)
        
        # Compute anomaly score based on spectral characteristics
        anomaly_score = self._compute_anomaly_score(radial_profile, high_freq_ratio)
        
        return {
            'magnitude_spectrum': magnitude_log.tolist(),
            'anomaly_score': float(anomaly_score),
            'radial_profile': radial_profile.tolist(),
            'high_freq_ratio': float(high_freq_ratio),
            'interpretation': self._interpret_score(anomaly_score)
        }
    
    def _compute_radial_profile(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Compute radial average of FFT magnitude
        
        This shows how energy is distributed across frequencies.
        Natural images typically follow a 1/f power law.
        """
        center = magnitude.shape[0] // 2
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
        
        # Bin by radius
        max_r = center
        radial_sum = np.bincount(r.ravel(), magnitude.ravel(), minlength=max_r)
        radial_count = np.bincount(r.ravel(), minlength=max_r)
        radial_count[radial_count == 0] = 1  # Avoid division by zero
        
        radial_profile = radial_sum[:max_r] / radial_count[:max_r]
        return radial_profile
    
    def _compute_frequency_ratio(self, magnitude: np.ndarray) -> float:
        """
        Compute ratio of high to low frequency energy
        
        Deepfakes often have unusual high-frequency content
        from generation artifacts or missing natural texture.
        """
        center = magnitude.shape[0] // 2
        
        # Low freq: inner 25% of spectrum
        low_freq_radius = center // 4
        # High freq: outer 25% of spectrum
        high_freq_inner = 3 * center // 4
        
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        low_mask = r <= low_freq_radius
        high_mask = r >= high_freq_inner
        
        low_energy = np.mean(magnitude[low_mask])
        high_energy = np.mean(magnitude[high_mask])
        
        # Ratio (normalized to avoid extreme values)
        if low_energy > 0:
            ratio = high_energy / low_energy
        else:
            ratio = 0
        
        return ratio
    
    def _compute_anomaly_score(self, radial_profile: np.ndarray, 
                                high_freq_ratio: float) -> float:
        """
        Compute overall anomaly score
        
        Combines multiple frequency-based indicators.
        """
        # Check for deviation from 1/f power law
        # Natural images: log(power) should decrease linearly with log(frequency)
        freqs = np.arange(1, len(radial_profile))
        log_freq = np.log(freqs)
        log_power = np.log(radial_profile[1:] + 1e-10)
        
        # Linear fit to log-log plot
        if len(log_freq) > 2:
            coeffs = np.polyfit(log_freq, log_power, 1)
            slope = coeffs[0]
            
            # Expected slope for natural images is around -1 to -2
            # Deviation from this range is suspicious
            slope_deviation = abs(slope + 1.5)  # Centered around -1.5
            slope_score = min(1.0, slope_deviation / 2.0)
        else:
            slope_score = 0.5
        
        # High frequency ratio score
        # Typical real images have ratio < 0.1
        # GAN artifacts often push this higher
        freq_score = min(1.0, high_freq_ratio / 0.2)
        
        # Combine scores (weighted average)
        anomaly_score = 0.6 * slope_score + 0.4 * freq_score
        
        return np.clip(anomaly_score, 0, 1)
    
    def _interpret_score(self, score: float) -> str:
        """Human-readable interpretation of anomaly score"""
        if score < 0.3:
            return "LOW - Frequency pattern appears natural"
        elif score < 0.6:
            return "MEDIUM - Some frequency anomalies detected"
        else:
            return "HIGH - Significant frequency artifacts detected"


class LandmarkAnalyzer:
    """
    Facial Landmark Consistency Analysis
    
    AI-generated faces often have subtly incorrect proportions:
    - Eyes at slightly wrong heights
    - Asymmetric ear positions
    - Unnatural nose-to-mouth ratios
    
    This analyzer computes 68 facial landmarks and checks
    geometric consistency against expected human proportions.
    """
    
    # Golden ratio and typical facial proportions
    EXPECTED_RATIOS = {
        'interocular_to_face_width': (0.25, 0.35),  # Eye separation / face width
        'nose_to_chin_ratio': (0.35, 0.45),          # Nose-chin / face height
        'eye_to_mouth_ratio': (0.35, 0.45),          # Eye-mouth / face height
        'symmetry_tolerance': 0.15,                   # Max asymmetry ratio
    }
    
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize landmark analyzer
        
        Args:
            predictor_path: Path to dlib shape predictor (optional)
                           If not provided, attempts to download or use mediapipe
        """
        self.backend = LANDMARK_BACKEND
        self.detector = None
        self.predictor = None
        
        if self.backend == "dlib":
            self._init_dlib(predictor_path)
        elif self.backend == "mediapipe":
            self._init_mediapipe()
        else:
            print("⚠️ No landmark detection backend available")
            print("   Install dlib: pip install dlib")
            print("   Or mediapipe: pip install mediapipe")
    
    def _init_dlib(self, predictor_path: Optional[str] = None):
        """Initialize dlib face detector and landmark predictor"""
        self.detector = dlib.get_frontal_face_detector()
        
        # Try to find predictor file
        if predictor_path and os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            # Common locations
            common_paths = [
                "shape_predictor_68_face_landmarks.dat",
                "./models/shape_predictor_68_face_landmarks.dat",
                os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat"),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    self.predictor = dlib.shape_predictor(path)
                    break
            
            if self.predictor is None:
                print("⚠️ dlib predictor not found. Download from:")
                print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("   Falling back to basic analysis")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def analyze(self, image_path: Union[str, Image.Image]) -> Dict:
        """
        Analyze facial landmark consistency
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Dictionary with:
            - landmarks: List of (x, y) coordinates
            - consistency_score: 0-1 score (higher = more consistent/natural)
            - symmetry_score: 0-1 score for facial symmetry
            - proportion_scores: Individual ratio scores
            - issues: List of detected anomalies
        """
        if self.backend is None:
            return {
                'error': 'No landmark detection backend available',
                'consistency_score': 0.5,
                'symmetry_score': 0.5,
                'issues': ['Landmark detection unavailable']
            }
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Get landmarks based on backend
        if self.backend == "dlib":
            landmarks = self._get_landmarks_dlib(image)
        else:
            landmarks = self._get_landmarks_mediapipe(image)
        
        if landmarks is None or len(landmarks) == 0:
            return {
                'error': 'No face detected',
                'consistency_score': 0.5,
                'symmetry_score': 0.5,
                'issues': ['No face detected in image']
            }
        
        # Analyze proportions
        proportion_scores = self._analyze_proportions(landmarks, image.shape)
        symmetry_score = self._analyze_symmetry(landmarks)
        
        # Compute overall consistency score
        issues = []
        all_scores = list(proportion_scores.values()) + [symmetry_score]
        consistency_score = np.mean(all_scores)
        
        # Flag specific issues
        if symmetry_score < 0.7:
            issues.append("Facial asymmetry detected")
        for name, score in proportion_scores.items():
            if score < 0.6:
                issues.append(f"Unusual {name.replace('_', ' ')}")
        
        return {
            'landmarks': landmarks.tolist() if isinstance(landmarks, np.ndarray) else landmarks,
            'consistency_score': float(consistency_score),
            'symmetry_score': float(symmetry_score),
            'proportion_scores': {k: float(v) for k, v in proportion_scores.items()},
            'issues': issues,
            'interpretation': self._interpret_score(consistency_score)
        }
    
    def _get_landmarks_dlib(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 68 facial landmarks using dlib"""
        if self.predictor is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Use first detected face
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        return landmarks
    
    def _get_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using MediaPipe"""
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert to pixel coordinates
        landmarks = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in face_landmarks.landmark
        ])
        
        # MediaPipe has 468 landmarks, map to approximate 68-point model
        # Key indices for 68-point compatibility
        key_indices = [
            # Jaw line (0-16)
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            # Left eyebrow (17-21)
            336, 296, 334, 293, 300,
            # Right eyebrow (22-26)
            107, 66, 105, 63, 70,
            # Nose bridge (27-30)
            168, 6, 197, 195,
            # Nose bottom (31-35)
            5, 4, 1, 19, 94,
            # Left eye (36-41)
            33, 246, 161, 160, 159, 158,
            # Right eye (42-47)
            263, 466, 388, 387, 386, 385,
            # Outer lips (48-59)
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
            # Inner lips (60-67)
            78, 191, 80, 81, 82, 13, 312, 311
        ]
        
        # Return subset of landmarks
        return landmarks[key_indices[:min(len(key_indices), len(landmarks))]]
    
    def _analyze_proportions(self, landmarks: np.ndarray, 
                             image_shape: Tuple) -> Dict[str, float]:
        """Analyze facial proportions against expected ratios"""
        scores = {}
        
        h, w = image_shape[:2]
        
        # Estimate face bounding box from landmarks
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        if face_width < 10 or face_height < 10:
            return {'overall': 0.5}
        
        # Interocular distance (approximate using landmark indices)
        if len(landmarks) >= 48:
            # Left eye center (landmarks 36-41), Right eye center (42-47)
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            interocular = np.linalg.norm(right_eye - left_eye)
            
            ratio = interocular / face_width
            expected = self.EXPECTED_RATIOS['interocular_to_face_width']
            scores['interocular_distance'] = self._ratio_score(ratio, expected)
        
        # Nose to chin ratio (if we have enough landmarks)
        if len(landmarks) >= 58:
            nose_tip = landmarks[30] if len(landmarks) > 30 else landmarks[-1]
            chin = landmarks[8] if len(landmarks) > 8 else landmarks[-1]
            nose_to_chin = np.linalg.norm(chin - nose_tip)
            
            ratio = nose_to_chin / face_height
            expected = self.EXPECTED_RATIOS['nose_to_chin_ratio']
            scores['nose_to_chin'] = self._ratio_score(ratio, expected)
        
        return scores if scores else {'overall': 0.7}
    
    def _analyze_symmetry(self, landmarks: np.ndarray) -> float:
        """Analyze facial symmetry"""
        if len(landmarks) < 48:
            return 0.7  # Default score if not enough landmarks
        
        # Compute center line of face
        center_x = landmarks[:, 0].mean()
        
        # Compare left vs right landmarks
        # Left eye (36-41) vs Right eye (42-47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Mirror right eye across center
        right_eye_mirrored = right_eye.copy()
        right_eye_mirrored[:, 0] = 2 * center_x - right_eye_mirrored[:, 0]
        
        # Compute average distance between mirrored points
        distances = np.linalg.norm(left_eye - right_eye_mirrored, axis=1)
        avg_distance = distances.mean()
        
        # Normalize by face size
        face_size = landmarks[:, 0].max() - landmarks[:, 0].min()
        normalized_distance = avg_distance / face_size if face_size > 0 else 0
        
        # Convert to score (lower distance = higher symmetry score)
        tolerance = self.EXPECTED_RATIOS['symmetry_tolerance']
        symmetry_score = max(0, 1 - (normalized_distance / tolerance))
        
        return symmetry_score
    
    def _ratio_score(self, actual: float, expected: Tuple[float, float]) -> float:
        """Score how well an actual ratio matches expected range"""
        min_val, max_val = expected
        center = (min_val + max_val) / 2
        range_half = (max_val - min_val) / 2
        
        if min_val <= actual <= max_val:
            return 1.0
        elif actual < min_val:
            deviation = (min_val - actual) / range_half
        else:
            deviation = (actual - max_val) / range_half
        
        return max(0, 1 - deviation)
    
    def _interpret_score(self, score: float) -> str:
        """Human-readable interpretation"""
        if score > 0.8:
            return "HIGH - Facial proportions appear natural"
        elif score > 0.6:
            return "MEDIUM - Minor geometric inconsistencies"
        else:
            return "LOW - Significant facial geometry anomalies"


class DeepfakePreprocessor:
    """
    Combined preprocessing pipeline for deepfake detection
    
    Runs both FFT and landmark analysis and combines results.
    """
    
    def __init__(self):
        self.fft_analyzer = FFTAnalyzer()
        self.landmark_analyzer = LandmarkAnalyzer()
    
    def analyze(self, image_path: Union[str, Image.Image]) -> Dict:
        """
        Run full preprocessing analysis
        
        Args:
            image_path: Path to image or PIL Image
            
        Returns:
            Combined analysis results with overall suspicion score
        """
        fft_result = self.fft_analyzer.analyze(image_path)
        landmark_result = self.landmark_analyzer.analyze(image_path)
        
        # Combine scores
        fft_score = fft_result.get('anomaly_score', 0.5)
        # Landmark consistency: higher = more natural, 
        # so invert for suspicion score
        landmark_score = 1 - landmark_result.get('consistency_score', 0.5)
        
        # Weighted combination
        combined_suspicion = 0.5 * fft_score + 0.5 * landmark_score
        
        return {
            'fft_analysis': fft_result,
            'landmark_analysis': landmark_result,
            'combined_suspicion_score': float(combined_suspicion),
            'interpretation': self._interpret_combined(combined_suspicion),
            'recommendation': self._get_recommendation(fft_score, landmark_score)
        }
    
    def _interpret_combined(self, score: float) -> str:
        """Interpret combined score"""
        if score < 0.3:
            return "LOW SUSPICION - Image appears authentic"
        elif score < 0.5:
            return "MODERATE SUSPICION - Some artifacts detected"
        elif score < 0.7:
            return "HIGH SUSPICION - Multiple anomalies present"
        else:
            return "VERY HIGH SUSPICION - Strong deepfake indicators"
    
    def _get_recommendation(self, fft_score: float, landmark_score: float) -> str:
        """Provide specific recommendation based on analysis"""
        if fft_score > 0.6 and landmark_score > 0.6:
            return "Both frequency and facial geometry show anomalies - likely fake"
        elif fft_score > 0.6:
            return "Frequency artifacts detected - possible AI generation/manipulation"
        elif landmark_score > 0.6:
            return "Facial proportions unusual - possible face swap or generation"
        else:
            return "No strong indicators of manipulation in preprocessing"


# Convenience functions
def analyze_fft(image_path: Union[str, Image.Image]) -> Dict:
    """Quick FFT analysis"""
    return FFTAnalyzer().analyze(image_path)


def analyze_landmarks(image_path: Union[str, Image.Image]) -> Dict:
    """Quick landmark analysis"""
    return LandmarkAnalyzer().analyze(image_path)


def full_preprocessing(image_path: Union[str, Image.Image]) -> Dict:
    """Run full preprocessing pipeline"""
    return DeepfakePreprocessor().analyze(image_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("DEEPFAKE PREPROCESSING ANALYZER")
        print("="*70)
        print("\nUsage:")
        print("  python preprocessing.py <image_path>")
        print("\nAnalyzes image using:")
        print("  • FFT (Frequency Domain) - Detects generation artifacts")
        print("  • Facial Landmarks - Checks geometric consistency")
        print("="*70 + "\n")
    else:
        image_path = sys.argv[1]
        print(f"\n🔍 Analyzing: {image_path}\n")
        
        result = full_preprocessing(image_path)
        
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

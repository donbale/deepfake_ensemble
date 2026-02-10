#!/usr/bin/env python3
"""
Face Forensics Analyzer - Mathematical face analysis for deepfake detection

Pure signal-processing approach (no ML model required):
1. FFT Frequency Analysis  - GAN/diffusion artifacts in frequency domain
2. Landmark Consistency    - Unnatural facial proportions via dlib 68-point
3. Symmetry Analysis       - Left/right face symmetry (GANs → too perfect)
4. Texture Consistency     - Local variance analysis for unnatural smoothness
5. Edge/Blending Artifacts - Gradient discontinuities at face boundary

Dependencies: numpy, opencv-python, dlib, Pillow
"""

import numpy as np
import cv2
from PIL import Image
import os
import warnings

# Try to import dlib for landmark analysis
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    warnings.warn("dlib not installed. Landmark/symmetry analysis disabled. Install with: pip install dlib")


class FaceForensicsAnalyzer:
    """
    Mathematical face forensics detector.
    
    Analyzes faces using signal processing techniques:
    - Frequency domain (FFT) for GAN artifacts
    - Facial landmark geometry for proportion consistency
    - Bilateral symmetry analysis
    - Texture variance analysis
    - Edge gradient analysis for blending boundaries
    
    Returns standardized output compatible with ensemble detector.
    """
    
    # Weights for combining sub-scores into final fake_probability
    SCORE_WEIGHTS = {
        'frequency': 0.30,
        'landmark': 0.20,
        'symmetry': 0.15,
        'texture': 0.20,
        'edge': 0.15,
    }
    
    # dlib shape predictor model filename
    SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
    
    def __init__(self, predictor_path=None, device='cpu'):
        """
        Initialize face forensics analyzer.
        
        Args:
            predictor_path: Path to dlib shape_predictor_68_face_landmarks.dat
                           If None, will look in ./models/dlib/ or download
            device: Ignored (CPU-only, kept for API compatibility)
        """
        self.device = device  # Ignored, kept for API compat
        self.face_detector = None
        self.landmark_predictor = None
        self.landmarks_available = False
        
        if DLIB_AVAILABLE:
            self._init_dlib(predictor_path)
        else:
            print("   ⚠️  dlib not available - landmark/symmetry analysis disabled")
            print("      Frequency + texture + edge analysis still active")
        
        print("✓ Face Forensics Analyzer ready")
    
    def _init_dlib(self, predictor_path):
        """Initialize dlib face detector and landmark predictor."""
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Find shape predictor file
        if predictor_path and os.path.exists(predictor_path):
            pred_path = predictor_path
        else:
            # Check common locations
            candidates = [
                os.path.join("models", "dlib", self.SHAPE_PREDICTOR),
                os.path.join(".", self.SHAPE_PREDICTOR),
                self.SHAPE_PREDICTOR,
            ]
            pred_path = None
            for c in candidates:
                if os.path.exists(c):
                    pred_path = c
                    break
            
            if pred_path is None:
                # Try to download
                pred_path = self._download_predictor()
        
        if pred_path and os.path.exists(pred_path):
            self.landmark_predictor = dlib.shape_predictor(pred_path)
            self.landmarks_available = True
            print(f"   ✓ Landmark predictor loaded: {pred_path}")
        else:
            print(f"   ⚠️  Shape predictor not found. Landmark analysis disabled.")
            print(f"      Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"      Extract to: models/dlib/")
    
    def _download_predictor(self):
        """Try to download the dlib shape predictor."""
        import urllib.request
        import bz2
        
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        dest_dir = os.path.join("models", "dlib")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, self.SHAPE_PREDICTOR)
        bz2_path = dest_path + ".bz2"
        
        try:
            print(f"   📥 Downloading shape predictor (~65MB)...")
            urllib.request.urlretrieve(url, bz2_path)
            
            print(f"   📦 Extracting...")
            with bz2.BZ2File(bz2_path) as fr, open(dest_path, 'wb') as fw:
                fw.write(fr.read())
            
            os.remove(bz2_path)
            print(f"   ✓ Saved to {dest_path}")
            return dest_path
        except Exception as e:
            print(f"   ⚠️  Download failed: {e}")
            return None
    
    def _load_image(self, image_path):
        """Load image as numpy array (BGR for OpenCV)."""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL as fallback
                pil_img = Image.open(image_path).convert('RGB')
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_path, Image.Image):
            img = np.array(image_path.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image_path
        return img
    
    def _detect_face(self, img):
        """Detect the largest face in the image, return bounding box."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.face_detector is not None:
            # Use dlib
            faces = self.face_detector(gray, 1)
            if faces:
                # Return largest face
                face = max(faces, key=lambda f: (f.right()-f.left()) * (f.bottom()-f.top()))
                return (face.left(), face.top(), face.right(), face.bottom()), gray
        
        # Fallback: use OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        if len(faces) > 0:
            # Largest face
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return (x, y, x+w, y+h), gray
        
        return None, gray
    
    def _get_landmarks(self, gray, face_rect):
        """Get 68 facial landmarks using dlib."""
        if not self.landmarks_available or self.landmark_predictor is None:
            return None
        
        x1, y1, x2, y2 = face_rect
        dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        shape = self.landmark_predictor(gray, dlib_rect)
        
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        return landmarks
    
    def _crop_face(self, img, face_rect, padding=0.2):
        """Crop face region with padding."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_rect
        fw, fh = x2 - x1, y2 - y1
        
        # Add padding
        px, py = int(fw * padding), int(fh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        
        return img[y1:y2, x1:x2]
    
    # ─── Analysis 1: Frequency Domain (FFT) ────────────────────────
    
    def analyze_frequency(self, face_crop):
        """
        Analyze frequency spectrum of face crop.
        
        GAN-generated images often show:
        - Unusual peaks in high-frequency bands
        - Grid-like patterns from upsampling
        - Abnormal energy distribution vs. natural images
        
        Returns: anomaly score 0-1 (higher = more suspicious)
        """
        # Convert to grayscale
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        # Resize to standard size for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        gray = gray.astype(np.float64)
        
        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log magnitude spectrum
        mag_log = np.log1p(magnitude)
        
        h, w = mag_log.shape
        cy, cx = h // 2, w // 2
        
        # Compute energy in frequency bands (radial)
        max_radius = min(cy, cx)
        n_bands = 8
        band_energy = []
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        for i in range(n_bands):
            r_inner = i * max_radius / n_bands
            r_outer = (i + 1) * max_radius / n_bands
            mask = (dist >= r_inner) & (dist < r_outer)
            band_e = np.mean(mag_log[mask]) if np.any(mask) else 0
            band_energy.append(band_e)
        
        band_energy = np.array(band_energy)
        
        # Score 1: High-to-low frequency energy ratio
        # Natural images have most energy in low frequencies
        # GAN images often have elevated high-frequency energy
        if band_energy[0] > 0:
            hf_ratio = np.mean(band_energy[5:]) / (np.mean(band_energy[:3]) + 1e-8)
        else:
            hf_ratio = 0.5
        
        # Score 2: Spectral flatness (how uniform the spectrum is)
        # GAN images tend to have flatter spectra
        if np.all(band_energy > 0):
            geo_mean = np.exp(np.mean(np.log(band_energy + 1e-8)))
            arith_mean = np.mean(band_energy)
            spectral_flatness = geo_mean / (arith_mean + 1e-8)
        else:
            spectral_flatness = 0.5
        
        # Score 3: Check for periodic artifacts (grid pattern from upsampling)
        # Look for peaks in the magnitude spectrum
        mag_normalized = mag_log / (np.max(mag_log) + 1e-8)
        peak_threshold = 0.7
        peaks = np.sum(mag_normalized > peak_threshold)
        total_pixels = h * w
        peak_ratio = peaks / total_pixels
        
        # Combine scores
        # Higher hf_ratio → more suspicious
        # Higher spectral_flatness → more suspicious  
        # Higher peak_ratio → could indicate artifacts
        anomaly_score = (
            0.4 * min(hf_ratio / 0.5, 1.0) +  # Normalize around 0.5 expected ratio
            0.35 * spectral_flatness +
            0.25 * min(peak_ratio / 0.01, 1.0)
        )
        
        return np.clip(anomaly_score, 0, 1)
    
    # ─── Analysis 2: Landmark Consistency ───────────────────────────
    
    def analyze_landmarks(self, landmarks):
        """
        Check facial landmark proportions against natural ranges.
        
        Uses known facial geometry ratios (golden ratio, etc.):
        - Inter-eye distance vs face width
        - Nose length vs face height
        - Mouth width vs inter-eye distance
        
        Returns: anomaly score 0-1
        """
        if landmarks is None:
            return 0.5  # Neutral if no landmarks
        
        # Key landmark indices (dlib 68-point):
        # 36-41: left eye, 42-47: right eye
        # 27-35: nose, 48-67: mouth
        # 0-16: jawline
        
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        nose_tip = landmarks[30]
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        chin = landmarks[8]
        forehead_approx = (landmarks[19] + landmarks[24]) / 2  # Between eyebrows
        
        # Distances
        inter_eye = np.linalg.norm(right_eye_center - left_eye_center)
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        face_height = np.linalg.norm(forehead_approx - chin)
        nose_length = np.linalg.norm(forehead_approx - nose_tip)
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        # Guard against degenerate detections
        if inter_eye < 5 or face_width < 10 or face_height < 10:
            return 0.5
        
        # Ratios (natural ranges based on anthropometric studies)
        ratios = {}
        anomalies = []
        
        # Eye distance / face width: typically 0.35-0.45
        r1 = inter_eye / face_width
        ratios['eye_face_width'] = r1
        anomalies.append(self._ratio_anomaly(r1, 0.35, 0.45))
        
        # Nose length / face height: typically 0.35-0.50
        r2 = nose_length / face_height
        ratios['nose_face_height'] = r2
        anomalies.append(self._ratio_anomaly(r2, 0.35, 0.50))
        
        # Mouth width / inter-eye distance: typically 0.8-1.2
        r3 = mouth_width / inter_eye
        ratios['mouth_eye_ratio'] = r3
        anomalies.append(self._ratio_anomaly(r3, 0.8, 1.2))
        
        # Face width / face height: typically 0.65-0.85
        r4 = face_width / face_height
        ratios['face_proportions'] = r4
        anomalies.append(self._ratio_anomaly(r4, 0.65, 0.85))
        
        # Average anomaly
        return np.clip(np.mean(anomalies), 0, 1)
    
    def _ratio_anomaly(self, value, low, high):
        """Score how anomalous a ratio is. 0 = within range, 1 = very anomalous."""
        if low <= value <= high:
            return 0.0
        
        mid = (low + high) / 2
        range_size = (high - low) / 2
        deviation = abs(value - mid) - range_size
        # Normalize: deviation of 1 range_size beyond limits → score 1.0
        return min(deviation / (range_size + 1e-8), 1.0)
    
    # ─── Analysis 3: Symmetry Analysis ──────────────────────────────
    
    def analyze_symmetry(self, face_crop, landmarks=None):
        """
        Analyze bilateral facial symmetry.
        
        GANs sometimes produce faces that are TOO symmetrical
        (unnatural perfection) or have asymmetric artifacts.
        
        Returns: anomaly score 0-1
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        h, w = gray.shape
        half_w = w // 2
        
        # Split face into left and right halves
        left_half = gray[:, :half_w]
        right_half = cv2.flip(gray[:, half_w:2*half_w], 1)  # Mirror right half
        
        # Ensure same size
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w].astype(np.float64)
        right_half = right_half[:, :min_w].astype(np.float64)
        
        if left_half.size == 0 or right_half.size == 0:
            return 0.5
        
        # Pixel-level symmetry difference
        diff = np.abs(left_half - right_half)
        mean_diff = np.mean(diff) / 255.0
        
        # Natural faces: mean_diff typically 0.05-0.15
        # Too low (<0.03) → suspiciously symmetric (GAN)
        # Too high (>0.20) → possible face swap artifacts
        
        if mean_diff < 0.03:
            # Too symmetric — suspicious for GAN
            anomaly = 0.3 + (0.03 - mean_diff) / 0.03 * 0.4
        elif mean_diff > 0.20:
            # Too asymmetric — possible blending artifact
            anomaly = 0.3 + (mean_diff - 0.20) / 0.20 * 0.4
        else:
            # Normal range
            anomaly = 0.1
        
        # Structural similarity (SSIM-like) at block level
        block_size = max(16, min(h, min_w) // 8)
        block_diffs = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, min_w - block_size, block_size):
                lb = left_half[y:y+block_size, x:x+block_size]
                rb = right_half[y:y+block_size, x:x+block_size]
                
                std_l = np.std(lb)
                std_r = np.std(rb)
                
                if std_l > 1 and std_r > 1:
                    corr = np.corrcoef(lb.flatten(), rb.flatten())[0, 1]
                    if not np.isnan(corr):
                        block_diffs.append(1 - abs(corr))
        
        if block_diffs:
            block_inconsistency = np.mean(block_diffs)
            anomaly = 0.5 * anomaly + 0.5 * block_inconsistency
        
        return np.clip(anomaly, 0, 1)
    
    # ─── Analysis 4: Texture Consistency ────────────────────────────
    
    def analyze_texture(self, face_crop):
        """
        Analyze texture consistency across the face.
        
        Deepfakes often show:
        - Unnaturally smooth skin (low local variance)
        - Inconsistent texture between regions (forehead vs cheek)
        - Repeated texture patterns
        
        Returns: anomaly score 0-1
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float64)
        
        # Local variance map
        kernel_size = 7
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance_map = sqr_mean - mean**2
        variance_map = np.maximum(variance_map, 0)  # Numerical stability
        
        # Score 1: Overall smoothness
        mean_var = np.mean(variance_map)
        # Natural faces: mean_var typically 200-800
        # Too smooth (<100) → suspicious
        smoothness_score = max(0, 1 - mean_var / 400)
        
        # Score 2: Variance consistency across regions
        h, w = gray.shape
        regions = [
            variance_map[:h//3, :],          # Forehead
            variance_map[h//3:2*h//3, :],    # Mid-face
            variance_map[2*h//3:, :],        # Lower face
        ]
        region_means = [np.mean(r) for r in regions]
        
        if max(region_means) > 0:
            region_cv = np.std(region_means) / (np.mean(region_means) + 1e-8)
        else:
            region_cv = 0.5
        
        # High region CV → inconsistent texture → suspicious
        texture_inconsistency = min(region_cv / 1.0, 1.0)
        
        # Score 3: Laplacian-based sharpness (blurriness detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        # Very low lap_var → blurry face (common in low-quality deepfakes)
        blur_score = max(0, 1 - lap_var / 500)
        
        anomaly = (
            0.35 * smoothness_score +
            0.35 * texture_inconsistency +
            0.30 * blur_score
        )
        
        return np.clip(anomaly, 0, 1)
    
    # ─── Analysis 5: Edge/Blending Artifacts ────────────────────────
    
    def analyze_edges(self, img, face_rect):
        """
        Analyze gradient discontinuities at the face boundary.
        
        Face swaps create blending artifacts at the boundary
        where the manipulated face meets the original background.
        
        Returns: anomaly score 0-1
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_rect
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img.astype(np.float64)
        
        # Create a band around the face boundary
        band_width = max(10, min(x2-x1, y2-y1) // 10)
        
        # Inner and outer regions along the boundary
        gradient_diffs = []
        
        # Sample points along face boundary
        boundary_points = []
        for x in range(max(0, x1), min(w, x2), band_width):
            boundary_points.append((x, y1))  # Top
            boundary_points.append((x, y2))  # Bottom
        for y in range(max(0, y1), min(h, y2), band_width):
            boundary_points.append((x1, y))  # Left
            boundary_points.append((x2, y))  # Right
        
        for bx, by in boundary_points:
            # Get gradient magnitude inside and outside the boundary
            inner_x = max(0, min(w-1-band_width, bx))
            inner_y = max(0, min(h-1-band_width, by))
            
            # Compute local gradient
            patch_inner = gray[
                max(0, inner_y-band_width):min(h, inner_y+band_width),
                max(0, inner_x-band_width):min(w, inner_x+band_width)
            ]
            
            if patch_inner.size > 0:
                gx = cv2.Sobel(patch_inner, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch_inner, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx**2 + gy**2)
                gradient_diffs.append(np.mean(grad_mag))
        
        if not gradient_diffs:
            return 0.5
        
        gradient_diffs = np.array(gradient_diffs)
        
        # Score: High and inconsistent gradients at boundary → suspicious
        mean_grad = np.mean(gradient_diffs)
        std_grad = np.std(gradient_diffs)
        cv_grad = std_grad / (mean_grad + 1e-8)
        
        # High mean gradient at boundary → blending artifacts
        grad_score = min(mean_grad / 50, 1.0)
        # High variability → inconsistent blending
        var_score = min(cv_grad / 1.5, 1.0)
        
        anomaly = 0.5 * grad_score + 0.5 * var_score
        return np.clip(anomaly, 0, 1)
    
    # ─── Main Prediction ───────────────────────────────────────────
    
    def predict(self, image_path, return_all_probs=False):
        """
        Run all forensic analyses and produce a fake probability.
        
        Returns:
            Dictionary with standardized prediction results
        """
        img = self._load_image(image_path)
        
        if img is None or img.size == 0:
            return {
                'predicted_class': 1,
                'predicted_label': 'Real',
                'confidence': 0.5,
                'is_fake': False,
                'fake_probability': 0.5,
            }
        
        # Detect face
        face_rect, gray = self._detect_face(img)
        
        if face_rect is None:
            # No face detected — return neutral
            return {
                'predicted_class': 1,
                'predicted_label': 'Real',
                'confidence': 0.5,
                'is_fake': False,
                'fake_probability': 0.5,
                'detail': 'No face detected',
            }
        
        # Crop face
        face_crop = self._crop_face(img, face_rect)
        
        # Get landmarks if available
        landmarks = self._get_landmarks(gray, face_rect)
        
        # Run all analyses
        scores = {}
        
        scores['frequency'] = self.analyze_frequency(face_crop)
        scores['texture'] = self.analyze_texture(face_crop)
        scores['edge'] = self.analyze_edges(img, face_rect)
        scores['symmetry'] = self.analyze_symmetry(face_crop, landmarks)
        
        if landmarks is not None:
            scores['landmark'] = self.analyze_landmarks(landmarks)
        else:
            # Redistribute landmark weight to other analyses
            scores['landmark'] = 0.5  # Neutral
        
        # Weighted combination
        fake_prob = sum(
            self.SCORE_WEIGHTS[k] * scores[k]
            for k in self.SCORE_WEIGHTS
        )
        fake_prob = np.clip(fake_prob, 0, 1)
        
        is_fake = fake_prob >= 0.5
        confidence = abs(fake_prob - 0.5) * 2  # 0→0.5 maps to 0→1
        
        result = {
            'predicted_class': 0 if is_fake else 1,
            'predicted_label': 'Fake' if is_fake else 'Real',
            'confidence': confidence,
            'is_fake': is_fake,
            'fake_probability': float(fake_prob),
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                'Fake': float(fake_prob),
                'Real': float(1 - fake_prob),
            }
            result['sub_scores'] = {k: float(v) for k, v in scores.items()}
        
        return result
    
    def predict_batch(self, image_paths):
        """Predict on multiple images."""
        return [self.predict(p, return_all_probs=True) for p in image_paths]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Forensics Analyzer')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--predictor', type=str, default=None,
                        help='Path to shape_predictor_68_face_landmarks.dat')
    
    args = parser.parse_args()
    
    print("\n🔬 Initializing Face Forensics Analyzer...")
    analyzer = FaceForensicsAnalyzer(predictor_path=args.predictor)
    
    print(f"\n🔍 Analyzing: {args.image}")
    result = analyzer.predict(args.image, return_all_probs=True)
    
    emoji = "🚨" if result['is_fake'] else "✅"
    print(f"\n{emoji} PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")
    print(f"🎯 FAKE PROBABILITY: {result['fake_probability']*100:.2f}%")
    
    if 'sub_scores' in result:
        print(f"\n📈 SUB-SCORES:")
        for name, score in result['sub_scores'].items():
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"  {name:12s} {bar} {score*100:.1f}%")

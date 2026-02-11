#!/usr/bin/env python3
"""
Face Forensics Analyzer v2 - Advanced mathematical face analysis for deepfake detection

Pure signal-processing approach (no ML model required):
1. Frequency Analysis (v2)  - DCT + azimuthal FFT averaging for GAN/diffusion artifacts
2. Landmark Consistency     - Facial proportions via dlib 68-point landmarks
3. Symmetry Analysis (v2)   - Landmark-aligned bilateral symmetry in LAB color space
4. Texture Analysis (v2)    - LBP histograms + local variance + Laplacian
5. Edge/Blending Artifacts  - Gradient discontinuities at face boundary
6. Noise Residual Analysis  - Sensor noise consistency (NEW)
7. Color Consistency        - White balance / illumination analysis (NEW)

Dependencies: numpy, opencv-python, dlib (optional), Pillow
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
    Advanced mathematical face forensics detector (v2).
    
    7 analysis modules combining signal processing, texture analysis,
    and color consistency checks. No ML models required.
    """
    
    # Weights for combining sub-scores into final fake_probability
    # 7 modules (sum = 1.0)
    SCORE_WEIGHTS = {
        'frequency': 0.20,
        'landmark': 0.10,
        'symmetry': 0.10,
        'texture': 0.15,
        'edge': 0.10,
        'noise': 0.20,
        'color': 0.15,
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
        self.device = device
        self.face_detector = None
        self.landmark_predictor = None
        self.landmarks_available = False
        
        if DLIB_AVAILABLE:
            self._init_dlib(predictor_path)
        else:
            print("   ⚠️  dlib not available - landmark/symmetry analysis disabled")
            print("      Frequency + texture + edge + noise + color analysis still active")
        
        print("✓ Face Forensics Analyzer v2 ready (7 modules)")
    
    def _init_dlib(self, predictor_path):
        """Initialize dlib face detector and landmark predictor."""
        self.face_detector = dlib.get_frontal_face_detector()
        
        if predictor_path and os.path.exists(predictor_path):
            pred_path = predictor_path
        else:
            candidates = [
                os.path.join("models", "dlib", self.SHAPE_PREDICTOR),
                os.path.join("/models", "dlib", self.SHAPE_PREDICTOR),
                os.path.join(".", self.SHAPE_PREDICTOR),
                self.SHAPE_PREDICTOR,
            ]
            pred_path = None
            for c in candidates:
                if os.path.exists(c):
                    pred_path = c
                    break
            
            if pred_path is None:
                pred_path = self._download_predictor()
        
        if pred_path and os.path.exists(pred_path):
            self.landmark_predictor = dlib.shape_predictor(pred_path)
            self.landmarks_available = True
            print(f"   ✓ Landmark predictor loaded: {pred_path}")
        else:
            print(f"   ⚠️  Shape predictor not found. Landmark analysis disabled.")
            print(f"      Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    
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
    
    # ─── Image / Face Utilities ─────────────────────────────────────
    
    def _load_image(self, image_path):
        """Load image as numpy array (BGR for OpenCV)."""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
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
        """Detect the largest face in the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.face_detector is not None:
            faces = self.face_detector(gray, 1)
            if faces:
                face = max(faces, key=lambda f: (f.right()-f.left()) * (f.bottom()-f.top()))
                return (face.left(), face.top(), face.right(), face.bottom()), gray
        
        # Fallback: OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        if len(faces) > 0:
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
        
        px, py = int(fw * padding), int(fh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        
        return img[y1:y2, x1:x2]
    
    def _align_face(self, face_crop, landmarks, face_rect):
        """Align face using eye landmarks for symmetry analysis."""
        if landmarks is None:
            return face_crop
        
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Compute rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate around midpoint between eyes
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Adjust center relative to face crop
        x1, y1, _, _ = face_rect
        crop_center = (eye_center[0] - x1, eye_center[1] - y1)
        
        h, w = face_crop.shape[:2]
        # Clamp center to valid range
        crop_center = (
            max(0, min(w, crop_center[0])),
            max(0, min(h, crop_center[1])),
        )
        
        M = cv2.getRotationMatrix2D(crop_center, angle, 1.0)
        aligned = cv2.warpAffine(face_crop, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        return aligned
    
    # ─── Analysis 1: Frequency Domain (DCT + Azimuthal FFT) ────────
    
    def analyze_frequency(self, face_crop):
        """
        v2: DCT block analysis + azimuthal FFT averaging.
        
        GAN artifacts show up as:
        - Abnormal DCT coefficient distributions in 8x8 blocks
        - Deviation from natural power-law spectral falloff
        - Grid-like patterns from upsampling (periodic peaks)
        
        Returns: anomaly score 0-1
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float64)
        
        # ── Part A: DCT block analysis ──
        dct_score = self._analyze_dct_blocks(gray)
        
        # ── Part B: Azimuthal FFT averaging ──
        azimuthal_score = self._analyze_azimuthal_spectrum(gray)
        
        # ── Part C: Periodic peak detection (original, improved) ──
        peak_score = self._analyze_spectral_peaks(gray)
        
        anomaly = 0.40 * dct_score + 0.35 * azimuthal_score + 0.25 * peak_score
        return np.clip(anomaly, 0, 1)
    
    def _analyze_dct_blocks(self, gray):
        """
        Analyze 8x8 DCT blocks like JPEG compression.
        GAN images have different DCT coefficient distributions than photos.
        """
        h, w = gray.shape
        block_size = 8
        
        # Collect AC coefficient statistics across all blocks
        ac_variances = []
        ac_kurtoses = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block)
                
                # Extract AC coefficients (skip DC at [0,0])
                ac_coeffs = dct_block.flatten()[1:]
                
                if len(ac_coeffs) > 0 and np.std(ac_coeffs) > 0:
                    ac_variances.append(np.var(ac_coeffs))
                    
                    # Kurtosis: natural images → higher kurtosis (heavy tails)
                    # GAN images → lower kurtosis (more Gaussian-like)
                    mean = np.mean(ac_coeffs)
                    std = np.std(ac_coeffs)
                    if std > 1e-6:
                        kurt = np.mean(((ac_coeffs - mean) / std) ** 4) - 3
                        ac_kurtoses.append(kurt)
        
        if not ac_variances:
            return 0.5
        
        # Natural images: high variance in AC coefficients, high kurtosis
        mean_var = np.mean(ac_variances)
        var_score = max(0, 1 - mean_var / 500)  # Low variance → suspicious
        
        if ac_kurtoses:
            mean_kurt = np.mean(ac_kurtoses)
            # Natural images typically have kurtosis > 3
            # GAN images closer to 0 (Gaussian)
            kurt_score = max(0, 1 - mean_kurt / 6)
        else:
            kurt_score = 0.5
        
        return 0.5 * var_score + 0.5 * kurt_score
    
    def _analyze_azimuthal_spectrum(self, gray):
        """
        Average FFT magnitude along circles (azimuthal averaging).
        Natural images: power-law falloff 1/f^alpha (alpha ~ 1.5-2.5).
        GAN images: deviate from this, especially at high frequencies.
        """
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        mag_log = np.log1p(magnitude)
        
        h, w = mag_log.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Compute azimuthal average (mean magnitude per radius)
        n_bins = 64
        radii = np.linspace(1, max_radius, n_bins)
        azimuthal_avg = []
        
        for i in range(len(radii) - 1):
            mask = (dist >= radii[i]) & (dist < radii[i+1])
            if np.any(mask):
                azimuthal_avg.append(np.mean(mag_log[mask]))
            else:
                azimuthal_avg.append(0)
        
        azimuthal_avg = np.array(azimuthal_avg)
        
        if len(azimuthal_avg) < 10 or azimuthal_avg[0] == 0:
            return 0.5
        
        # Fit power law: log(power) = -alpha * log(freq) + const
        # Use log-log linear regression
        valid = azimuthal_avg > 0
        if np.sum(valid) < 5:
            return 0.5
        
        log_freq = np.log(radii[:-1][valid])
        log_power = np.log(azimuthal_avg[valid])
        
        # Linear fit
        A = np.vstack([log_freq, np.ones(len(log_freq))]).T
        try:
            result = np.linalg.lstsq(A, log_power, rcond=None)
            slope = result[0][0]
            residuals = log_power - A @ result[0]
            fit_error = np.std(residuals)
        except np.linalg.LinAlgError:
            return 0.5
        
        alpha = -slope  # Power law exponent
        
        # Natural images: alpha ~ 1.5-2.5, low fit error
        # GAN images: alpha outside this range or high fit error
        
        alpha_score = 0.0
        if alpha < 1.2 or alpha > 3.0:
            alpha_score = min(abs(alpha - 2.0) / 2.0, 1.0)
        elif alpha < 1.5 or alpha > 2.5:
            alpha_score = min(abs(alpha - 2.0) / 2.0, 0.5)
        
        # High residuals → doesn't follow natural power law
        error_score = min(fit_error / 0.5, 1.0)
        
        return 0.5 * alpha_score + 0.5 * error_score
    
    def _analyze_spectral_peaks(self, gray):
        """Detect periodic artifacts (grid patterns from upsampling)."""
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        mag_log = np.log1p(magnitude)
        
        mag_normalized = mag_log / (np.max(mag_log) + 1e-8)
        
        # Exclude DC component region
        h, w = mag_normalized.shape
        cy, cx = h // 2, w // 2
        mask = np.ones_like(mag_normalized, dtype=bool)
        Y, X = np.ogrid[:h, :w]
        dc_region = ((X - cx)**2 + (Y - cy)**2) < 25  # radius 5 around DC
        mask[dc_region] = False
        
        peak_threshold = 0.6
        peaks = np.sum(mag_normalized[mask] > peak_threshold)
        total_pixels = np.sum(mask)
        peak_ratio = peaks / (total_pixels + 1e-8)
        
        return min(peak_ratio / 0.005, 1.0)
    
    # ─── Analysis 2: Landmark Consistency ───────────────────────────
    
    def analyze_landmarks(self, landmarks):
        """
        Check facial landmark proportions against natural ranges.
        
        Uses anthropometric ratios:
        - Inter-eye distance vs face width
        - Nose length vs face height  
        - Mouth width vs inter-eye distance
        - Eye aspect ratios (left vs right)
        - Philtrum / nose-to-lip ratio
        - Jawline curvature
        
        Returns: anomaly score 0-1
        """
        if landmarks is None:
            return 0.5
        
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        nose_tip = landmarks[30]
        nose_bridge = landmarks[27]
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        upper_lip_top = landmarks[51]
        chin = landmarks[8]
        forehead_approx = (landmarks[19] + landmarks[24]) / 2
        
        inter_eye = np.linalg.norm(right_eye_center - left_eye_center)
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        face_height = np.linalg.norm(forehead_approx - chin)
        nose_length = np.linalg.norm(forehead_approx - nose_tip)
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        if inter_eye < 5 or face_width < 10 or face_height < 10:
            return 0.5
        
        anomalies = []
        
        # Original ratios
        anomalies.append(self._ratio_anomaly(inter_eye / face_width, 0.35, 0.45))
        anomalies.append(self._ratio_anomaly(nose_length / face_height, 0.35, 0.50))
        anomalies.append(self._ratio_anomaly(mouth_width / inter_eye, 0.8, 1.2))
        anomalies.append(self._ratio_anomaly(face_width / face_height, 0.65, 0.85))
        
        # NEW: Eye aspect ratios (EAR) - left vs right should be similar
        left_ear = self._eye_aspect_ratio(landmarks[36:42])
        right_ear = self._eye_aspect_ratio(landmarks[42:48])
        if left_ear > 0 and right_ear > 0:
            ear_diff = abs(left_ear - right_ear) / max(left_ear, right_ear)
            anomalies.append(min(ear_diff / 0.3, 1.0))
        
        # NEW: Philtrum ratio (nose tip to upper lip vs nose length)
        philtrum_length = np.linalg.norm(upper_lip_top - nose_tip)
        nose_bridge_length = np.linalg.norm(nose_tip - nose_bridge)
        if nose_bridge_length > 5:
            anomalies.append(self._ratio_anomaly(philtrum_length / nose_bridge_length, 0.3, 0.7))
        
        # NEW: Jawline curvature smoothness
        jaw_score = self._jawline_smoothness(landmarks[0:17])
        anomalies.append(jaw_score)
        
        return np.clip(np.mean(anomalies), 0, 1)
    
    def _eye_aspect_ratio(self, eye_pts):
        """Compute eye aspect ratio from 6 landmarks."""
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])
        if h < 1:
            return 0
        return (v1 + v2) / (2.0 * h)
    
    def _jawline_smoothness(self, jaw_pts):
        """
        Check jawline for unnaturally smooth or jagged contours.
        Deepfakes often have irregularities in the jawline.
        """
        if len(jaw_pts) < 5:
            return 0.5
        
        # Compute angles between consecutive jawline segments
        angles = []
        for i in range(1, len(jaw_pts) - 1):
            v1 = jaw_pts[i] - jaw_pts[i-1]
            v2 = jaw_pts[i+1] - jaw_pts[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1 or norm2 < 1:
                continue
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)
        
        if not angles:
            return 0.5
        
        # Natural jawline: smooth with gentle angle changes
        # Deepfake: either too smooth (low std) or jagged (high std)
        angle_std = np.std(angles)
        
        if angle_std < 3:
            # Too smooth
            return 0.3
        elif angle_std > 25:
            # Too jagged
            return min(angle_std / 40, 1.0)
        else:
            return 0.05
    
    def _ratio_anomaly(self, value, low, high):
        """Score how anomalous a ratio is. 0 = within range, 1 = very anomalous."""
        if low <= value <= high:
            return 0.0
        mid = (low + high) / 2
        range_size = (high - low) / 2
        deviation = abs(value - mid) - range_size
        return min(deviation / (range_size + 1e-8), 1.0)
    
    # ─── Analysis 3: Symmetry Analysis (v2 - aligned, LAB) ─────────
    
    def analyze_symmetry(self, face_crop, landmarks=None, face_rect=None):
        """
        v2: Landmark-aligned bilateral symmetry in LAB color space.
        
        1. Aligns face using eye landmarks (removes rotation bias)
        2. Analyzes symmetry in L, A, B channels separately
        3. Color bleed detection in A/B channels
        
        Returns: anomaly score 0-1
        """
        # Align face if landmarks available
        if landmarks is not None and face_rect is not None:
            aligned = self._align_face(face_crop, landmarks, face_rect)
        else:
            aligned = face_crop
        
        # Convert to LAB for multi-channel analysis
        if len(aligned.shape) == 3:
            lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        else:
            # Grayscale fallback
            lab = np.stack([aligned, aligned, aligned], axis=2)
        
        h, w = lab.shape[:2]
        half_w = w // 2
        
        if half_w < 5 or h < 10:
            return 0.5
        
        channel_scores = []
        
        for ch in range(3):
            channel = lab[:, :, ch].astype(np.float64)
            left_half = channel[:, :half_w]
            right_half = cv2.flip(channel[:, half_w:2*half_w], 1)
            
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            
            if left_half.size == 0 or right_half.size == 0:
                channel_scores.append(0.5)
                continue
            
            # Pixel difference
            diff = np.abs(left_half - right_half)
            
            if ch == 0:
                # L channel: normalize to 0-1 (L ranges 0-255 in OpenCV LAB)
                mean_diff = np.mean(diff) / 255.0
            else:
                # A, B channels: centered at 128, range ~80-180
                mean_diff = np.mean(diff) / 128.0
            
            # Scoring
            if ch == 0:
                # L (luminance): same logic as before
                if mean_diff < 0.03:
                    score = 0.3 + (0.03 - mean_diff) / 0.03 * 0.4
                elif mean_diff > 0.20:
                    score = 0.3 + (mean_diff - 0.20) / 0.20 * 0.4
                else:
                    score = 0.1
            else:
                # A, B (color): asymmetry in color channels = blending artifact
                # Natural faces: mean_diff ~ 0.01-0.05
                if mean_diff > 0.08:
                    score = min(0.3 + (mean_diff - 0.08) / 0.1, 1.0)
                elif mean_diff < 0.005:
                    score = 0.2  # Suspiciously uniform color
                else:
                    score = 0.1
            
            channel_scores.append(score)
        
        # Block-level structural consistency (on luminance)
        gray = lab[:, :, 0].astype(np.float64)
        left_g = gray[:, :half_w]
        right_g = cv2.flip(gray[:, half_w:2*half_w], 1)
        min_w = min(left_g.shape[1], right_g.shape[1])
        
        block_size = max(16, min(h, min_w) // 8)
        block_diffs = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, min_w - block_size, block_size):
                lb = left_g[y:y+block_size, x:x+block_size]
                rb = right_g[y:y+block_size, x:x+block_size]
                
                std_l = np.std(lb)
                std_r = np.std(rb)
                
                if std_l > 1 and std_r > 1:
                    corr = np.corrcoef(lb.flatten(), rb.flatten())[0, 1]
                    if not np.isnan(corr):
                        block_diffs.append(1 - abs(corr))
        
        block_score = np.mean(block_diffs) if block_diffs else 0.3
        
        # Combine: L channel (0.3), A channel (0.15), B channel (0.15), blocks (0.4)
        anomaly = (
            0.30 * channel_scores[0] +
            0.15 * channel_scores[1] +
            0.15 * channel_scores[2] +
            0.40 * block_score
        )
        
        return np.clip(anomaly, 0, 1)
    
    # ─── Analysis 4: Texture Analysis (v2 - LBP + variance) ────────
    
    def analyze_texture(self, face_crop):
        """
        v2: LBP histogram analysis + local variance + Laplacian.
        
        LBP (Local Binary Patterns) capture micro-texture patterns.
        GAN-generated faces produce measurably different LBP distributions.
        
        Returns: anomaly score 0-1
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float64)
        
        # ── Part A: LBP histogram analysis ──
        lbp_score = self._analyze_lbp(gray)
        
        # ── Part B: Local variance (original, retained) ──
        variance_score = self._analyze_local_variance(gray)
        
        # ── Part C: Laplacian sharpness ──
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        blur_score = max(0, 1 - lap_var / 500)
        
        anomaly = 0.40 * lbp_score + 0.35 * variance_score + 0.25 * blur_score
        return np.clip(anomaly, 0, 1)
    
    def _analyze_lbp(self, gray):
        """
        Compute LBP (Local Binary Pattern) histogram and analyze distribution.
        Real faces have characteristic LBP distributions that GANs don't match.
        """
        gray_uint8 = gray.astype(np.uint8)
        h, w = gray_uint8.shape
        
        # Compute LBP (simplified uniform LBP)
        lbp = np.zeros_like(gray_uint8)
        
        for dy, dx, bit in [(-1,-1,0), (-1,0,1), (-1,1,2), (0,1,3),
                             (1,1,4), (1,0,5), (1,-1,6), (0,-1,7)]:
            y_start = max(0, dy)
            y_end = h + min(0, dy)
            x_start = max(0, dx)
            x_end = w + min(0, dx)
            
            cy_start = max(0, -dy)
            cy_end = h + min(0, -dy)
            cx_start = max(0, -dx)
            cx_end = w + min(0, -dx)
            
            neighbor = gray_uint8[y_start:y_end, x_start:x_end]
            center = gray_uint8[cy_start:cy_end, cx_start:cx_end]
            
            min_h = min(neighbor.shape[0], center.shape[0])
            min_w = min(neighbor.shape[1], center.shape[1])
            
            lbp_region = lbp[cy_start:cy_start+min_h, cx_start:cx_start+min_w]
            lbp_region |= ((neighbor[:min_h, :min_w] >= center[:min_h, :min_w]).astype(np.uint8) << bit)
        
        # Compute histogram
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256), density=True)
        hist = hist + 1e-10  # Avoid log(0)
        
        # Metric 1: Entropy — natural textures have moderate entropy
        # Too high (uniform random) or too low (very regular) = suspicious
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(256)
        norm_entropy = entropy / max_entropy
        
        # Natural faces: entropy typically 0.55-0.80
        if norm_entropy > 0.85:
            entropy_score = min((norm_entropy - 0.85) / 0.15, 1.0)
        elif norm_entropy < 0.45:
            entropy_score = min((0.45 - norm_entropy) / 0.2, 1.0)
        else:
            entropy_score = 0.05
        
        # Metric 2: Uniformity — sum of squared probabilities
        # Higher uniformity → fewer dominant patterns → possibly GAN
        uniformity = np.sum(hist ** 2)
        # Natural range: 0.01-0.05
        if uniformity > 0.08:
            unif_score = min((uniformity - 0.08) / 0.05, 1.0)
        elif uniformity < 0.008:
            unif_score = 0.4  # Too uniform = flat texture
        else:
            unif_score = 0.05
        
        # Metric 3: Regional LBP consistency
        # Split into quadrants and compare histograms
        quad_hists = []
        qh, qw = h // 2, w // 2
        for qy in [0, qh]:
            for qx in [0, qw]:
                quad = lbp[qy:qy+qh, qx:qx+qw]
                qhist, _ = np.histogram(quad.flatten(), bins=256, range=(0, 256), density=True)
                quad_hists.append(qhist)
        
        # Chi-squared distances between quadrants
        chi_dists = []
        for i in range(len(quad_hists)):
            for j in range(i+1, len(quad_hists)):
                chi = np.sum((quad_hists[i] - quad_hists[j])**2 / (quad_hists[i] + quad_hists[j] + 1e-10))
                chi_dists.append(chi)
        
        # Very similar quadrants → possibly synthetic
        mean_chi = np.mean(chi_dists) if chi_dists else 0.5
        if mean_chi < 0.02:
            chi_score = 0.4  # Suspiciously uniform
        elif mean_chi > 0.3:
            chi_score = min(mean_chi / 0.5, 1.0)  # Very different regions
        else:
            chi_score = 0.05
        
        return 0.4 * entropy_score + 0.3 * unif_score + 0.3 * chi_score
    
    def _analyze_local_variance(self, gray):
        """Local variance analysis for unnatural smoothness."""
        kernel_size = 7
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance_map = np.maximum(sqr_mean - mean**2, 0)
        
        mean_var = np.mean(variance_map)
        smoothness_score = max(0, 1 - mean_var / 400)
        
        h, w = gray.shape
        regions = [
            variance_map[:h//3, :],
            variance_map[h//3:2*h//3, :],
            variance_map[2*h//3:, :],
        ]
        region_means = [np.mean(r) for r in regions]
        
        if max(region_means) > 0:
            region_cv = np.std(region_means) / (np.mean(region_means) + 1e-8)
        else:
            region_cv = 0.5
        
        inconsistency = min(region_cv / 1.0, 1.0)
        
        return 0.5 * smoothness_score + 0.5 * inconsistency
    
    # ─── Analysis 5: Edge/Blending Artifacts ────────────────────────
    
    def analyze_edges(self, img, face_rect):
        """
        Analyze gradient discontinuities at the face boundary.
        Face swaps create blending artifacts where the altered face
        meets the original background.
        
        Returns: anomaly score 0-1
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_rect
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img.astype(np.float64)
        
        band_width = max(10, min(x2-x1, y2-y1) // 10)
        
        # Sample points along boundary
        boundary_points = []
        for x in range(max(0, x1), min(w, x2), band_width):
            boundary_points.append((x, y1))
            boundary_points.append((x, y2))
        for y in range(max(0, y1), min(h, y2), band_width):
            boundary_points.append((x1, y))
            boundary_points.append((x2, y))
        
        gradient_diffs = []
        for bx, by in boundary_points:
            inner_x = max(0, min(w-1-band_width, bx))
            inner_y = max(0, min(h-1-band_width, by))
            
            patch = gray[
                max(0, inner_y-band_width):min(h, inner_y+band_width),
                max(0, inner_x-band_width):min(w, inner_x+band_width)
            ]
            
            if patch.size > 0:
                gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx**2 + gy**2)
                gradient_diffs.append(np.mean(grad_mag))
        
        if not gradient_diffs:
            return 0.5
        
        gradient_diffs = np.array(gradient_diffs)
        mean_grad = np.mean(gradient_diffs)
        std_grad = np.std(gradient_diffs)
        cv_grad = std_grad / (mean_grad + 1e-8)
        
        grad_score = min(mean_grad / 50, 1.0)
        var_score = min(cv_grad / 1.5, 1.0)
        
        # NEW: Color channel edge analysis
        color_edge_score = 0.5
        if len(img.shape) == 3:
            color_edge_score = self._analyze_color_edges(img, face_rect, band_width)
        
        anomaly = 0.35 * grad_score + 0.35 * var_score + 0.30 * color_edge_score
        return np.clip(anomaly, 0, 1)
    
    def _analyze_color_edges(self, img, face_rect, band_width):
        """Check for blending artifacts visible in individual color channels."""
        x1, y1, x2, y2 = face_rect
        h, w = img.shape[:2]
        
        channel_gradients = []
        for ch in range(3):
            channel = img[:, :, ch].astype(np.float64)
            
            # Get mean gradient at boundary for this channel
            grads = []
            for x in range(max(0, x1), min(w-1, x2), band_width * 2):
                for y_pos in [y1, y2]:
                    if 0 < y_pos < h - 1:
                        patch = channel[
                            max(0, y_pos-band_width):min(h, y_pos+band_width),
                            max(0, x-band_width):min(w, x+band_width)
                        ]
                        if patch.size > 0:
                            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                            grads.append(np.mean(np.sqrt(gx**2 + gy**2)))
            
            if grads:
                channel_gradients.append(np.mean(grads))
        
        if len(channel_gradients) < 3:
            return 0.5
        
        # If one channel has significantly different gradient → blending artifact
        cg = np.array(channel_gradients)
        if np.mean(cg) > 0:
            channel_cv = np.std(cg) / (np.mean(cg) + 1e-8)
            return min(channel_cv / 0.5, 1.0)
        
        return 0.5
    
    # ─── Analysis 6: Noise Residual Analysis (NEW) ──────────────────
    
    def analyze_noise(self, face_crop):
        """
        Analyze sensor noise consistency.
        
        Real photos have consistent sensor noise from the camera.
        GAN/diffusion images lack this natural noise or have synthetic noise.
        Face swaps show noise inconsistency between pasted face and background.
        
        Method: subtract denoised image to extract noise residual,
        then analyze its statistics.
        
        Returns: anomaly score 0-1
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float64)
        
        # Extract noise residual: original - denoised
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise = gray - denoised
        
        # ── Metric 1: Noise level (std of residual) ──
        noise_std = np.std(noise)
        # Natural photos: noise_std typically 3-15
        # GAN images: often < 2 (too clean) or > 20 (added noise)
        if noise_std < 2:
            level_score = 0.5 + (2 - noise_std) / 4  # Too clean
        elif noise_std > 15:
            level_score = min(0.3 + (noise_std - 15) / 20, 1.0)  # Too noisy
        else:
            level_score = 0.05  # Normal range
        
        # ── Metric 2: Noise spatial consistency ──
        # Divide into blocks and compare noise levels
        block_size = 32
        h, w = noise.shape
        block_stds = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                block_stds.append(np.std(block))
        
        if block_stds:
            block_stds = np.array(block_stds)
            noise_cv = np.std(block_stds) / (np.mean(block_stds) + 1e-8)
            # High CV → inconsistent noise → face swap
            consistency_score = min(noise_cv / 0.8, 1.0)
        else:
            consistency_score = 0.5
        
        # ── Metric 3: Noise histogram normality ──
        # Natural sensor noise → approximately Gaussian
        # GAN noise → often non-Gaussian
        noise_flat = noise.flatten()
        noise_mean = np.mean(noise_flat)
        noise_s = np.std(noise_flat)
        
        if noise_s > 0.5:
            # Skewness
            skewness = np.mean(((noise_flat - noise_mean) / noise_s) ** 3)
            # Kurtosis (excess)
            kurtosis = np.mean(((noise_flat - noise_mean) / noise_s) ** 4) - 3
            
            # Gaussian: skew ≈ 0, kurtosis ≈ 0
            # Non-Gaussian → suspicious
            normality_score = min(
                (abs(skewness) / 1.0 + abs(kurtosis) / 3.0) / 2,
                1.0
            )
        else:
            normality_score = 0.6  # Too little noise to analyze
        
        # ── Metric 4: Noise in high-frequency band ──
        # Apply high-pass filter and check noise in that band
        laplacian_noise = cv2.Laplacian(gray, cv2.CV_64F)
        lap_std = np.std(laplacian_noise)
        
        # Ratio of high-freq noise to overall noise
        if noise_std > 0.5:
            hf_ratio = lap_std / (noise_std * 10 + 1e-8)
            # Natural: hf_ratio ~ 0.5-2.0
            if hf_ratio < 0.3:
                hf_score = 0.4
            elif hf_ratio > 3.0:
                hf_score = min(hf_ratio / 5.0, 1.0)
            else:
                hf_score = 0.05
        else:
            hf_score = 0.5
        
        anomaly = (
            0.25 * level_score +
            0.30 * consistency_score +
            0.25 * normality_score +
            0.20 * hf_score
        )
        
        return np.clip(anomaly, 0, 1)
    
    # ─── Analysis 7: Color Consistency (NEW) ────────────────────────
    
    def analyze_color(self, img, face_rect):
        """
        Check for color/illumination inconsistencies.
        
        Face swaps often have:
        - Different white balance between pasted face and background
        - Illumination direction mismatch
        - Skin tone that doesn't match the neck/ears
        
        Returns: anomaly score 0-1
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_rect
        
        if len(img.shape) < 3:
            return 0.5  # Need color image
        
        # Convert to LAB for perceptually uniform color analysis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        
        # Face region
        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(w, x2)
        fy2 = min(h, y2)
        face_lab = lab[fy1:fy2, fx1:fx2]
        
        if face_lab.size == 0:
            return 0.5
        
        # Background region (area around the face)
        pad = max(20, (x2 - x1) // 3)
        bg_regions = []
        
        # Above face
        if y1 > pad:
            bg_regions.append(lab[max(0, y1-pad):y1, fx1:fx2])
        # Below face
        if y2 + pad < h:
            bg_regions.append(lab[y2:min(h, y2+pad), fx1:fx2])
        # Left of face
        if x1 > pad:
            bg_regions.append(lab[fy1:fy2, max(0, x1-pad):x1])
        # Right of face
        if x2 + pad < w:
            bg_regions.append(lab[fy1:fy2, x2:min(w, x2+pad)])
        
        if not bg_regions:
            return 0.5
        
        bg_lab = np.vstack([r.reshape(-1, 3) for r in bg_regions if r.size >= 3])
        if bg_lab.shape[0] < 10:
            return 0.5
        
        face_flat = face_lab.reshape(-1, 3)
        
        # ── Metric 1: White balance difference (A, B channel means) ──
        face_a_mean = np.mean(face_flat[:, 1])
        face_b_mean = np.mean(face_flat[:, 2])
        bg_a_mean = np.mean(bg_lab[:, 1])
        bg_b_mean = np.mean(bg_lab[:, 2])
        
        # Color temperature difference
        delta_a = abs(face_a_mean - bg_a_mean)
        delta_b = abs(face_b_mean - bg_b_mean)
        color_diff = np.sqrt(delta_a**2 + delta_b**2)
        
        # Natural: some difference is expected (skin vs background)
        # But large differences in A/B channels → different lighting/camera
        wb_score = min(max(0, color_diff - 10) / 20, 1.0)
        
        # ── Metric 2: Luminance gradient direction ──
        # Check if lighting comes from the same direction on face and surroundings
        face_L = face_lab[:, :, 0]
        
        # Split face into quadrants and compare mean luminance
        fh, fw = face_L.shape
        if fh > 4 and fw > 4:
            top_left = np.mean(face_L[:fh//2, :fw//2])
            top_right = np.mean(face_L[:fh//2, fw//2:])
            bot_left = np.mean(face_L[fh//2:, :fw//2])
            bot_right = np.mean(face_L[fh//2:, fw//2:])
            
            # Compute gradient direction
            lr_diff = (top_right + bot_right) - (top_left + bot_left)
            tb_diff = (bot_left + bot_right) - (top_left + top_right)
            face_light_angle = np.arctan2(tb_diff, lr_diff)
            
            # Same analysis on the surrounding region
            full_L = lab[:, :, 0]
            sh, sw = full_L.shape
            surrounding_quadrants = [
                np.mean(full_L[:sh//2, :sw//2]),
                np.mean(full_L[:sh//2, sw//2:]),
                np.mean(full_L[sh//2:, :sw//2]),
                np.mean(full_L[sh//2:, sw//2:]),
            ]
            s_lr = (surrounding_quadrants[1] + surrounding_quadrants[3]) - \
                   (surrounding_quadrants[0] + surrounding_quadrants[2])
            s_tb = (surrounding_quadrants[2] + surrounding_quadrants[3]) - \
                   (surrounding_quadrants[0] + surrounding_quadrants[1])
            surr_light_angle = np.arctan2(s_tb, s_lr)
            
            angle_diff = abs(face_light_angle - surr_light_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            # Large angle difference → mismatched lighting
            light_score = min(angle_diff / (np.pi / 2), 1.0)
        else:
            light_score = 0.5
        
        # ── Metric 3: Skin tone consistency (face boundary) ──
        # Compare color at face edge vs just outside
        boundary_band = max(5, min(fx2-fx1, fy2-fy1) // 15)
        
        # Inner boundary strip
        inner_strips = []
        if fy2 - fy1 > 2 * boundary_band:
            inner_strips.append(face_lab[:boundary_band, :])  # Top inner
            inner_strips.append(face_lab[-boundary_band:, :])  # Bottom inner
        
        if inner_strips:
            inner = np.vstack([s.reshape(-1, 3) for s in inner_strips])
            inner_color = np.mean(inner, axis=0)
            bg_color = np.mean(bg_lab, axis=0)
            
            # Delta E (CIE76 approximation)
            delta_e = np.sqrt(np.sum((inner_color - bg_color) ** 2))
            # Large Delta E at boundary → potential splice
            skin_score = min(max(0, delta_e - 15) / 30, 1.0)
        else:
            skin_score = 0.5
        
        anomaly = 0.35 * wb_score + 0.30 * light_score + 0.35 * skin_score
        return np.clip(anomaly, 0, 1)
    
    # ─── Main Prediction ───────────────────────────────────────────
    
    def predict(self, image_path, return_all_probs=False):
        """
        Run all 7 forensic analyses and produce a fake probability.
        
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
        
        face_rect, gray = self._detect_face(img)
        
        if face_rect is None:
            return {
                'predicted_class': 1,
                'predicted_label': 'Real',
                'confidence': 0.5,
                'is_fake': False,
                'fake_probability': 0.5,
                'detail': 'No face detected',
            }
        
        face_crop = self._crop_face(img, face_rect)
        landmarks = self._get_landmarks(gray, face_rect)
        
        # Run all 7 analyses
        scores = {}
        
        # Original modules (upgraded)
        scores['frequency'] = self.analyze_frequency(face_crop)
        scores['texture'] = self.analyze_texture(face_crop)
        scores['edge'] = self.analyze_edges(img, face_rect)
        scores['symmetry'] = self.analyze_symmetry(face_crop, landmarks, face_rect)
        
        if landmarks is not None:
            scores['landmark'] = self.analyze_landmarks(landmarks)
        else:
            scores['landmark'] = 0.5  # Neutral without landmarks
        
        # NEW modules
        scores['noise'] = self.analyze_noise(face_crop)
        scores['color'] = self.analyze_color(img, face_rect)
        
        # Weighted combination
        fake_prob = sum(
            self.SCORE_WEIGHTS[k] * scores[k]
            for k in self.SCORE_WEIGHTS
        )
        fake_prob = np.clip(fake_prob, 0, 1)
        
        is_fake = fake_prob >= 0.5
        confidence = abs(fake_prob - 0.5) * 2
        
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
    
    parser = argparse.ArgumentParser(description='Face Forensics Analyzer v2')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--predictor', type=str, default=None,
                        help='Path to shape_predictor_68_face_landmarks.dat')
    
    args = parser.parse_args()
    
    print("\n🔬 Initializing Face Forensics Analyzer v2...")
    analyzer = FaceForensicsAnalyzer(predictor_path=args.predictor)
    
    print(f"\n🔍 Analyzing: {args.image}")
    result = analyzer.predict(args.image, return_all_probs=True)
    
    emoji = "🚨" if result['is_fake'] else "✅"
    print(f"\n{emoji} PREDICTION: {result['predicted_label']}")
    print(f"📊 CONFIDENCE: {result['confidence']*100:.2f}%")
    print(f"🎯 FAKE PROBABILITY: {result['fake_probability']*100:.2f}%")
    
    if 'sub_scores' in result:
        print(f"\n📈 SUB-SCORES (7 modules):")
        for name, score in result['sub_scores'].items():
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"  {name:12s} {bar} {score*100:.1f}%")

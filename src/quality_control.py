"""
Quality control: blur rejection, transition filtering, deduplication.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a frame."""
    frame_id: int
    laplacian_variance: float
    avg_intensity: float
    intensity_std: float
    is_sharp: bool
    is_transition: bool
    passes_quality: bool
    rejection_reason: str = ""


class QualityControl:
    """
    Three-stage quality control:
    1. Blur rejection (Laplacian variance)
    2. Transition filtering (fade detection)
    3. Deduplication (perceptual hash)
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        fade_black_threshold: float = 25.0,
        fade_white_threshold: float = 230.0,
        fade_std_threshold: float = 15.0,
        dedup_hash_distance: int = 5
    ):
        """
        Args:
            blur_threshold: Minimum Laplacian variance (higher = sharper required)
            fade_black_threshold: Max intensity for black fade
            fade_white_threshold: Min intensity for white fade
            fade_std_threshold: Max std for uniform fade
            dedup_hash_distance: Max pHash distance for duplicates
        """
        self.blur_threshold = blur_threshold
        self.fade_black_threshold = fade_black_threshold
        self.fade_white_threshold = fade_white_threshold
        self.fade_std_threshold = fade_std_threshold
        self.dedup_hash_distance = dedup_hash_distance
    
    def calculate_blur_score(self, frame: np.ndarray) -> float:
        """
        Calculate Laplacian variance (blur metric).
        
        Higher = sharper image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    def check_fade_transition(self, frame: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if frame is a fade transition.
        
        Returns:
            (is_fade, avg_intensity, intensity_std)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_intensity = gray.mean()
        intensity_std = gray.std()
        
        # Fade to black
        if avg_intensity < self.fade_black_threshold and intensity_std < self.fade_std_threshold:
            return True, float(avg_intensity), float(intensity_std)
        
        # Fade to white
        if avg_intensity > self.fade_white_threshold and intensity_std < self.fade_std_threshold:
            return True, float(avg_intensity), float(intensity_std)
        
        # Near-uniform frame
        if intensity_std < self.fade_std_threshold / 2:
            return True, float(avg_intensity), float(intensity_std)
        
        return False, float(avg_intensity), float(intensity_std)
    
    def evaluate_frame(self, frame: np.ndarray, frame_id: int) -> QualityMetrics:
        """
        Evaluate single frame quality.
        
        Returns:
            QualityMetrics with all checks
        """
        # Blur check
        blur_score = self.calculate_blur_score(frame)
        is_sharp = blur_score >= self.blur_threshold
        
        # Transition check
        is_fade, avg_intensity, intensity_std = self.check_fade_transition(frame)
        
        # Overall pass/fail
        passes = is_sharp and not is_fade
        
        reason = ""
        if not is_sharp:
            reason = f"blur (score={blur_score:.1f} < {self.blur_threshold})"
        elif is_fade:
            reason = f"fade/transition (intensity={avg_intensity:.1f}, std={intensity_std:.1f})"
        
        return QualityMetrics(
            frame_id=frame_id,
            laplacian_variance=blur_score,
            avg_intensity=avg_intensity,
            intensity_std=intensity_std,
            is_sharp=is_sharp,
            is_transition=is_fade,
            passes_quality=passes,
            rejection_reason=reason
        )
    
    def compute_phash(self, frame: np.ndarray) -> str:
        """
        Compute perceptual hash for deduplication.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Compute pHash
        hash_value = imagehash.phash(pil_image, hash_size=8)
        return str(hash_value)
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between hashes.
        """
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def deduplicate_frames(
        self,
        frames: List[np.ndarray],
        frame_ids: List[int]
    ) -> Tuple[List[int], List[str]]:
        """
        Remove duplicate frames based on perceptual hash.
        
        Returns:
            (kept_frame_indices, duplicate_pairs)
        """
        if not frames:
            return [], []
        
        logger.info(f"Deduplicating {len(frames)} frames...")
        
        # Compute hashes
        hashes = [self.compute_phash(frame) for frame in frames]
        
        # Track which frames to keep
        kept_indices = [0]  # Always keep first
        duplicates = []
        
        for i in range(1, len(frames)):
            is_duplicate = False
            
            for kept_idx in kept_indices:
                distance = self.hamming_distance(hashes[i], hashes[kept_idx])
                
                if distance <= self.dedup_hash_distance:
                    is_duplicate = True
                    duplicates.append(f"Frame {frame_ids[i]} duplicate of {frame_ids[kept_idx]} (distance={distance})")
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
        
        logger.info(f"✓ Kept {len(kept_indices)}/{len(frames)} frames ({len(frames)-len(kept_indices)} duplicates)")
        
        return kept_indices, duplicates


if __name__ == "__main__":
    # Quick test
    qc = QualityControl(blur_threshold=100.0, dedup_hash_distance=5)
    
    # Test with a sample frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    metrics = qc.evaluate_frame(test_frame, 0)
    
    print(f"Blur score: {metrics.laplacian_variance:.2f}")
    print(f"Is sharp: {metrics.is_sharp}")
    print(f"Passes quality: {metrics.passes_quality}")
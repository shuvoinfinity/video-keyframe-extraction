"""
Main pipeline: Scene detection → Quality control → Output
"""

import cv2
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import logging
import time

from .scene_detector import SceneDetector, AdaptiveSceneDetector, Scene
from .quality_control import QualityControl, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics from processing a video."""
    video_path: str
    video_duration: float
    processing_time: float
    
    # Scene detection
    scenes_detected: int
    avg_scene_duration: float
    detection_threshold: float
    
    # Quality control
    frames_extracted: int
    frames_blur_rejected: int
    frames_transition_rejected: int
    frames_dedup_removed: int
    frames_final: int
    
    # Quality metrics
    avg_blur_score: float
    min_blur_score: float
    max_blur_score: float


class KeyframeExtractionPipeline:
    """
    Complete pipeline for keyframe extraction.
    
    Steps:
    1. Scene detection (PySceneDetect, threshold=27)
    2. Keyframe extraction (middle frame of each scene)
    3. Quality control:
       - Blur rejection (Laplacian > 100)
       - Transition filtering (fade detection)
       - Deduplication (pHash distance < 5)
    4. Save keyframes and report
    """
    
    def __init__(
        self,
        output_dir: str = "data/output",
        scene_threshold: float = 27.0,
        blur_threshold: float = 100.0,
        dedup_hash_distance: int = 5,
        adaptive_detection: bool = False,
        target_scenes: int = 40
    ):
        """
        Args:
            output_dir: Where to save keyframes and reports
            scene_threshold: PySceneDetect threshold
            blur_threshold: Minimum blur score
            dedup_hash_distance: Max hash distance for duplicates
            adaptive_detection: Use adaptive threshold tuning
            target_scenes: Target scene count for adaptive mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        if adaptive_detection:
            self.scene_detector = AdaptiveSceneDetector(target_scenes=target_scenes)
        else:
            self.scene_detector = SceneDetector(threshold=scene_threshold)
        
        self.quality_control = QualityControl(
            blur_threshold=blur_threshold,
            dedup_hash_distance=dedup_hash_distance
        )
    
    def process_video(self, video_path: str, video_id: str = None) -> ProcessingStats:
        """
        Process a video through the complete pipeline.
        
        Args:
            video_path: Path to input video
            video_id: Unique identifier (uses filename if None)
        
        Returns:
            ProcessingStats with detailed metrics
        """
        start_time = time.time()
        
        if video_id is None:
            video_id = Path(video_path).stem
        
        logger.info("="*60)
        logger.info(f"Processing: {video_id}")
        logger.info("="*60)
        
        # Create output directories
        keyframe_dir = self.output_dir / "keyframes" / video_id
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        video_info = self._get_video_info(video_path)
        
        # STEP 1: Scene Detection
        logger.info("\n[STEP 1] Scene Detection...")
        scenes, scene_metadata = self.scene_detector.detect_scenes(video_path)
        
        if not scenes:
            logger.error("No scenes detected!")
            return None
        
        # STEP 2: Extract Keyframes
        logger.info("\n[STEP 2] Extracting Keyframes...")
        frames, frame_scenes = self._extract_keyframes(video_path, scenes)
        logger.info(f"✓ Extracted {len(frames)} keyframes")
        
        # STEP 3: Quality Control
        logger.info("\n[STEP 3] Quality Control...")
        
        # 3a: Blur and transition filtering
        quality_passed_frames = []
        quality_passed_scenes = []
        quality_metrics_list = []
        blur_rejected = 0
        transition_rejected = 0
        
        for frame, scene in zip(frames, frame_scenes):
            metrics = self.quality_control.evaluate_frame(frame, scene.scene_id)
            quality_metrics_list.append(metrics)
            
            if not metrics.is_sharp:
                blur_rejected += 1
                logger.debug(f"  Rejected scene {scene.scene_id}: {metrics.rejection_reason}")
                continue
            
            if metrics.is_transition:
                transition_rejected += 1
                logger.debug(f"  Rejected scene {scene.scene_id}: {metrics.rejection_reason}")
                continue
            
            quality_passed_frames.append(frame)
            quality_passed_scenes.append(scene)
        
        logger.info(f"✓ Blur rejected: {blur_rejected}")
        logger.info(f"✓ Transition rejected: {transition_rejected}")
        logger.info(f"✓ Quality passed: {len(quality_passed_frames)}")
        
        # 3b: Deduplication
        if len(quality_passed_frames) > 1:
            kept_indices, duplicate_info = self.quality_control.deduplicate_frames(
                quality_passed_frames,
                [s.scene_id for s in quality_passed_scenes]
            )
            
            final_frames = [quality_passed_frames[i] for i in kept_indices]
            final_scenes = [quality_passed_scenes[i] for i in kept_indices]
            dedup_removed = len(quality_passed_frames) - len(final_frames)
        else:
            final_frames = quality_passed_frames
            final_scenes = quality_passed_scenes
            dedup_removed = 0
            duplicate_info = []
        
        logger.info(f"✓ Duplicates removed: {dedup_removed}")
        logger.info(f"✓ Final keyframes: {len(final_frames)}")
        
        # STEP 4: Save Keyframes
        logger.info("\n[STEP 4] Saving Keyframes...")
        saved_paths = self._save_keyframes(final_frames, final_scenes, keyframe_dir)
        logger.info(f"✓ Saved to: {keyframe_dir}")
        
        # STEP 5: Generate Report
        processing_time = time.time() - start_time
        
        # Calculate statistics
        blur_scores = [m.laplacian_variance for m in quality_metrics_list if m.passes_quality]
        
        stats = ProcessingStats(
            video_path=video_path,
            video_duration=video_info['duration'],
            processing_time=processing_time,
            scenes_detected=len(scenes),
            avg_scene_duration=sum(s.duration for s in scenes) / len(scenes),
            detection_threshold=scene_metadata.get('threshold_used', scene_metadata.get('final_threshold', 27.0)),
            frames_extracted=len(frames),
            frames_blur_rejected=blur_rejected,
            frames_transition_rejected=transition_rejected,
            frames_dedup_removed=dedup_removed,
            frames_final=len(final_frames),
            avg_blur_score=sum(blur_scores) / len(blur_scores) if blur_scores else 0,
            min_blur_score=min(blur_scores) if blur_scores else 0,
            max_blur_score=max(blur_scores) if blur_scores else 0
        )
        
        # Save detailed report
        report = {
            'video_id': video_id,
            'stats': asdict(stats),
            'scene_metadata': scene_metadata,
            'quality_metrics': [asdict(m) for m in quality_metrics_list],
            'duplicate_info': duplicate_info,
            'final_keyframes': [
                {
                    'scene_id': scene.scene_id,
                    'timestamp': scene.start_time,
                    'duration': scene.duration,
                    'path': str(path)
                }
                for scene, path in zip(final_scenes, saved_paths)
            ]
        }
        
        report_path = report_dir / f"{video_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Report saved to: {report_path}")
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def _extract_keyframes(
        self,
        video_path: str,
        scenes: List[Scene]
    ) -> tuple[List, List[Scene]]:
        """Extract middle frame from each scene."""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        valid_scenes = []
        
        for scene in scenes:
            # Extract middle frame
            frame_num = scene.mid_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                valid_scenes.append(scene)
        
        cap.release()
        return frames, valid_scenes
    
    def _save_keyframes(
        self,
        frames: List,
        scenes: List[Scene],
        output_dir: Path
    ) -> List[Path]:
        """Save keyframes to disk."""
        saved_paths = []
        
        for frame, scene in zip(frames, scenes):
            filename = f"scene_{scene.scene_id:04d}_t{scene.start_time:.2f}s.jpg"
            filepath = output_dir / filename
            
            # Save with high quality
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(filepath)
        
        return saved_paths
    
    def _print_summary(self, stats: ProcessingStats):
        """Print processing summary."""
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Video duration:        {stats.video_duration:.2f}s")
        logger.info(f"Processing time:       {stats.processing_time:.2f}s")
        logger.info(f"")
        logger.info(f"Scenes detected:       {stats.scenes_detected}")
        logger.info(f"Detection threshold:   {stats.detection_threshold:.2f}")
        logger.info(f"Avg scene duration:    {stats.avg_scene_duration:.2f}s")
        logger.info(f"")
        logger.info(f"Frames extracted:      {stats.frames_extracted}")
        logger.info(f"Blur rejected:         {stats.frames_blur_rejected} ({stats.frames_blur_rejected/stats.frames_extracted*100:.1f}%)")
        logger.info(f"Transition rejected:   {stats.frames_transition_rejected} ({stats.frames_transition_rejected/stats.frames_extracted*100:.1f}%)")
        logger.info(f"Duplicates removed:    {stats.frames_dedup_removed} ({stats.frames_dedup_removed/stats.frames_extracted*100:.1f}%)")
        logger.info(f"")
        logger.info(f"FINAL KEYFRAMES:       {stats.frames_final}")
        logger.info(f"")
        logger.info(f"Blur scores:           min={stats.min_blur_score:.1f}, avg={stats.avg_blur_score:.1f}, max={stats.max_blur_score:.1f}")
        logger.info("="*60 + "\n")


if __name__ == "__main__":
    # Quick test
    pipeline = KeyframeExtractionPipeline(
        scene_threshold=27.0,
        blur_threshold=100.0,
        dedup_hash_distance=5
    )
    
    stats = pipeline.process_video("tests/sample_video.mp4", "sample")
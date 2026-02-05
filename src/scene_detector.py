"""
Scene detection using PySceneDetect with adaptive threshold tuning.
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass
from scenedetect import open_video, SceneManager, ContentDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Represents a detected scene."""
    scene_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def mid_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2


class SceneDetector:
    """
    Scene detection with configurable threshold.
    
    Default threshold=27 works well for most content.
    """
    
    def __init__(self, threshold: float = 27.0, min_scene_length: int = 15):
        """
        Args:
            threshold: Content detection threshold (15-40 typical range)
            min_scene_length: Minimum scene length in frames
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
    
    def detect_scenes(self, video_path: str) -> Tuple[List[Scene], dict]:
        """
        Detect scenes in video.
        
        Returns:
            (scenes, metadata)
        """
        logger.info(f"Detecting scenes in: {video_path}")
        logger.info(f"Threshold: {self.threshold}, Min scene length: {self.min_scene_length}")
        
        # Open video
        video = open_video(video_path)
        fps = video.frame_rate
        
        # Create scene manager
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_length
            )
        )
        
        # Detect scenes
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        # Convert to Scene objects
        scenes = []
        for idx, (start, end) in enumerate(scene_list):
            scene = Scene(
                scene_id=idx,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                start_frame=start.get_frames(),
                end_frame=end.get_frames()
            )
            scenes.append(scene)
        
        metadata = {
            'total_scenes': len(scenes),
            'fps': fps,
            'threshold_used': self.threshold
        }
        
        logger.info(f"✓ Detected {len(scenes)} scenes")
        return scenes, metadata


class AdaptiveSceneDetector:
    """
    Automatically tune threshold to achieve target scene count.
    """
    
    def __init__(self, target_scenes: int = 40, tolerance: int = 5):
        self.target_scenes = target_scenes
        self.tolerance = tolerance
    
    def detect_scenes(self, video_path: str) -> Tuple[List[Scene], dict]:
        """
        Detect scenes with adaptive threshold tuning.
        """
        low = 15.0
        high = 50.0
        best_result = None
        best_diff = float('inf')
        iterations_data = []
        
        logger.info(f"Adaptive detection targeting {self.target_scenes} scenes")
        
        for iteration in range(7):
            threshold = (low + high) / 2
            detector = SceneDetector(threshold=threshold)
            scenes, metadata = detector.detect_scenes(video_path)
            
            scene_count = len(scenes)
            diff = abs(scene_count - self.target_scenes)
            
            iterations_data.append({
                'iteration': iteration,
                'threshold': threshold,
                'scenes': scene_count,
                'diff': diff
            })
            
            logger.info(f"Iteration {iteration}: threshold={threshold:.2f} → {scene_count} scenes")
            
            if diff < best_diff:
                best_diff = diff
                best_result = (scenes, threshold)
            
            if diff <= self.tolerance:
                logger.info(f"✓ Converged at threshold {threshold:.2f}")
                break
            
            if scene_count < self.target_scenes:
                high = threshold
            else:
                low = threshold
        
        scenes, final_threshold = best_result
        metadata = {
            'total_scenes': len(scenes),
            'target_scenes': self.target_scenes,
            'final_threshold': final_threshold,
            'iterations': iterations_data
        }
        
        return scenes, metadata


if __name__ == "__main__":
    # Quick test
    detector = SceneDetector(threshold=27.0)
    scenes, meta = detector.detect_scenes("tests/sample_video.mp4")
    print(f"Detected {len(scenes)} scenes")
    for scene in scenes[:5]:
        print(f"  Scene {scene.scene_id}: {scene.start_time:.2f}s - {scene.end_time:.2f}s")
# Proposed Changes: Scene Detection Fallback

**File:** `src/scene_detector.py`

---

### Change 1: Implementation of Fallback Logic
**Location:** Lines 75-76 (inserted after `scene_list = scene_manager.get_scene_list()`)

**Current Code:**
```python
74:        scene_manager.detect_scenes(video, show_progress=True)
75:        scene_list = scene_manager.get_scene_list()
76:        
77:        # Convert to Scene objects
```

**Proposed Replacement:**
```python
74:        scene_manager.detect_scenes(video, show_progress=True)
75:        scene_list = scene_manager.get_scene_list()
76:        
77:        # --- FALLBACK LOGIC START ---
78:        if not scene_list:
79:            logger.warning("No scenes detected. Engaging 'Full Video' fallback.")
80:            from scenedetect import FrameTimecode
81:            
82:            # Get video end time
83:            end_frame = video.duration.get_frames()
84:            
85:            # Create a synthetic scene covering 0 to end
86:            scene_list = [(
87:                FrameTimecode(0, fps=video.frame_rate),
88:                FrameTimecode(end_frame, fps=video.frame_rate)
89:            )]
90:            fallback_used = True
91:        else:
92:            fallback_used = False
93:        # --- FALLBACK LOGIC END ---
94:
95:        # Convert to Scene objects
```

---

### Change 2: Metadata Update
**Location:** Lines 92-96 (updated to include `fallback_mode`)

**Current Code:**
```python
92:        metadata = {
93:            'total_scenes': len(scenes),
94:            'fps': fps,
95:            'threshold_used': self.threshold
96:        }
```

**Proposed Replacement:**
```python
106:        metadata = {
107:            'total_scenes': len(scenes),
108:            'fps': fps,
109:            'threshold_used': self.threshold,
110:            'fallback_mode': fallback_used
111:        }
```
*(Note: Line numbers shift due to Change 1 insertion)*

"""
Visualization utilities for testing and debugging.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
import json


def create_contact_sheet(image_paths: List[Path], output_path: str, cols: int = 5):
    """
    Create a contact sheet (grid) of all keyframes.
    
    Useful for visual inspection of results.
    """
    if not image_paths:
        print("No images to create contact sheet")
        return
    
    # Load all images and resize to same dimensions
    target_size = (320, 180)  # 16:9 aspect ratio
    images = []
    
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            # Add frame number text
            cv2.putText(
                img_resized,
                f"{path.stem}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            images.append(img_resized)
    
    if not images:
        print("No valid images loaded")
        return
    
    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    
    # Create blank canvas
    canvas_height = rows * target_size[1]
    canvas_width = cols * target_size[0]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place images on canvas
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        y_start = row * target_size[1]
        x_start = col * target_size[0]
        
        canvas[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = img
    
    # Save contact sheet
    cv2.imwrite(output_path, canvas)
    print(f"✓ Contact sheet saved to: {output_path}")


def generate_html_report(report_json_path: str, output_html_path: str):
    """
    Generate an HTML report for easy viewing.
    """
    with open(report_json_path, 'r') as f:
        report = json.load(f)
    
    stats = report['stats']
    keyframes = report['final_keyframes']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keyframe Extraction Report - {report['video_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #333; }}
            .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
            .stat-box {{ background: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
            .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .keyframes {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-top: 30px; }}
            .keyframe {{ background: #fff; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
            .keyframe img {{ width: 100%; height: auto; }}
            .keyframe-info {{ padding: 10px; font-size: 12px; }}
            .success {{ color: #4CAF50; }}
            .warning {{ color: #ff9800; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Keyframe Extraction Report</h1>
            <h2>{report['video_id']}</h2>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Scenes Detected</div>
                    <div class="stat-value">{stats['scenes_detected']}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Final Keyframes</div>
                    <div class="stat-value success">{stats['frames_final']}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Processing Time</div>
                    <div class="stat-value">{stats['processing_time']:.2f}s</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Blur Rejected</div>
                    <div class="stat-value warning">{stats['frames_blur_rejected']}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Transition Rejected</div>
                    <div class="stat-value warning">{stats['frames_transition_rejected']}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Duplicates Removed</div>
                    <div class="stat-value warning">{stats['frames_dedup_removed']}</div>
                </div>
            </div>
            
            <h3>Extracted Keyframes</h3>
            <div class="keyframes">
    """
    
    for kf in keyframes:
        # Use relative path
        img_path = Path(kf['path'])
        html += f"""
                <div class="keyframe">
                    <img src="{img_path.name}" alt="Scene {kf['scene_id']}">
                    <div class="keyframe-info">
                        <strong>Scene {kf['scene_id']}</strong><br>
                        Time: {kf['timestamp']:.2f}s<br>
                        Duration: {kf['duration']:.2f}s
                    </div>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML report saved to: {output_html_path}")


if __name__ == "__main__":
    # Test with sample data
    from glob import glob
    
    image_paths = [Path(p) for p in glob("data/output/keyframes/sample/*.jpg")]
    if image_paths:
        create_contact_sheet(image_paths, "data/output/contact_sheet.jpg")
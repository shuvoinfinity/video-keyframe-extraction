"""
Main entry point for keyframe extraction.

Usage:
    python run.py <video_path> [video_id]
    python run.py tests/sample_video.mp4 sample_test
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import KeyframeExtractionPipeline
from src.visualizer import create_contact_sheet, generate_html_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <video_path> [video_id]")
        print("Example: python run.py tests/sample_video.mp4 sample")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create pipeline with default settings
    pipeline = KeyframeExtractionPipeline(
        output_dir="data/output",
        scene_threshold=27.0,
        blur_threshold=5.0,
        dedup_hash_distance=5,
        adaptive_detection=True  # Set to True for adaptive threshold tuning
    )
    
    # Process video
    stats = pipeline.process_video(video_path, video_id)
    
    if stats:
        # Generate visualizations
        if video_id is None:
            video_id = Path(video_path).stem
        
        keyframe_dir = Path("data/output/keyframes") / video_id
        image_paths = sorted(keyframe_dir.glob("*.jpg"))
        
        if image_paths:
            # Create contact sheet
            contact_sheet_path = f"data/output/{video_id}_contact_sheet.jpg"
            create_contact_sheet(image_paths, contact_sheet_path)
            
            # Generate HTML report
            report_path = f"data/output/reports/{video_id}_report.json"
            html_path = f"data/output/{video_id}_report.html"
            generate_html_report(report_path, html_path)
            
            print(f"\n✓ Processing complete!")
            print(f"✓ Keyframes: {keyframe_dir}")
            print(f"✓ Contact sheet: {contact_sheet_path}")
            print(f"✓ HTML report: {html_path}")


if __name__ == "__main__":
    main()
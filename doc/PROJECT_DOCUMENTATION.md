# Video Keyframe Extraction Pipeline - Technical Documentation

**Date:** February 05, 2026  
**Project:** Video Keyframe Extraction  
**Language:** Python 3.10+

---

## 1. Executive Overview

### What does this project do?
This software is an automated "editor" that watches a video file and picks out the best static images (keyframes) to represent that video. Instead of just taking a screenshot every 10 seconds, it uses computer vision to:
1.  **Find natural cuts** (scenes) in the video.
2.  **Filter out bad images** (blurry, dark, or white frames).
3.  **Remove duplicates** (images that look too similar to ones already kept).
4.  **Create a report**, including a gallery website (HTML) and a single summary image (Contact Sheet).

### Target Audience for this Document
This guide is written for developers and non-experts alike. It explains *how* the code works, step-by-step, translating technical logic into plain English concepts.

---

## 2. System Architecture & Dependencies

### Module Dependencies
The project relies on a "Pipeline" architecture. Data flows linearly from one module to the next.

```text
run.py (Entry Point)
   │
   ▼
src/pipeline.py (The Manager)
   │
   ├── Uses: src/scene_detector.py (The Eyes) -> Finds where scenes start/end
   │
   ├── Uses: src/quality_control.py (The Gatekeeper) -> Checks blur & duplicates
   │
   └── Uses: src/visualizer.py (The Reporter) -> Draws images & writes HTML
```

### External Libraries: Roles & Responsibilities

This project is built on the shoulders of giants. Here is the specific role each library plays:

1.  **OpenCV (`cv2`): The "Eye and Hand"**
    *   **Role:** Raw visual input and output.
    *   **Responsibilities:** Reading the video file frame-by-frame, converting color spaces (Blue-Green-Red to Grayscale), resizing images, and writing the final JPG files to disk. It handles the low-level pixel manipulation.

2.  **PySceneDetect: The "Brain" (Temporal)**
    *   **Role:** Understanding time and change.
    *   **Responsibilities:** Scanning the video stream to calculate the difference between consecutive frames. It decides where a "shot" begins and ends based on visual discontinuities (cuts).

3.  **NumPy: The "Calculator"**
    *   **Role:** High-speed mathematics.
    *   **Responsibilities:** Images are just grids of numbers (matrices). NumPy performs the heavy mathematical lifting (calculating averages, standard deviations, variances) needed for blur detection and fade detection instantly.

4.  **ImageHash: The "Fingerprinter"**
    *   **Role:** Identification and memory.
    *   **Responsibilities:** It converts a complex image into a simple alphanumeric string (hash). This allows the system to compare two images ("Are these the same?") by comparing their strings rather than every single pixel, which is much faster.

5.  **Pillow (PIL): The "Canvas"**
    *   **Role:** Image composition.
    *   **Responsibilities:** Used primarily during report generation to handle image formats and creating the Contact Sheet grid.

---

## 3. The Analogy: Keyframe Detection and Marginal Error

One of the most difficult challenges in video analysis is the concept of "Marginal Error" in scene detection.

### The Problem
When does a scene *exactly* start? In a hard cut, it's instant. But in a cross-fade (where one scene dissolves into another), the "start" is fuzzy. If our detector is off by 5 frames (the **margin of error**), we might accidentally capture the blurry transition instead of the clear scene.

### The Solution: "Mid-Frame" Mitigation
To handle this marginal error, this pipeline uses a **Mid-Frame Selection Strategy**.
*   **Concept:** Instead of trying to grab the *first* frame of a scene (which has a high risk of being part of the transition), we calculate the exact mathematical center of the scene.
*   **Why it works:** Even if our cut detection has a margin of error of ±10 frames, the middle of the scene is statistically the safest place to be. It is the point furthest away from the dangerous "edges" (transitions).

### Adaptive Detection: Minimizing the Error Gap
We also use **Adaptive Thresholding** to minimize the error between *expected* output and *actual* output.
*   **Analogy:** Imagine tuning an old radio. You want to find clear stations.
    *   If you turn the dial too fast (High Threshold), you miss weaker stations (Under-detection).
    *   If you turn it too slow (Low Threshold), you stop at static noise (Over-detection).
*   **The Algorithm:** The `AdaptiveSceneDetector` automatically turns the dial back and forth (Binary Search) until the number of detected scenes matches your target (e.g., 40 scenes). It actively reduces the "error margin" of the result count.

---

## 4. Industry Standard Comparison

How does this Python script compare to what **Netflix, YouTube, or AWS** uses?

| Feature | This Project (Lite Pipeline) | Big Tech (Industry Standard) |
| :--- | :--- | :--- |
| **Scene Detection** | **Heuristic (PySceneDetect):** Uses pixel intensity differences. Fast and effective for clear cuts. | **Deep Learning (SBD):** Uses Neural Networks to understand "semantic" scene changes, even if the pixels change slowly (e.g., a slow pan). |
| **Quality Control** | **Laplacian Variance:** A simple mathematical check for "edginess" to detect blur. | **VMAF / BRISQUE:** Complex perceptual metrics that mimic how the human eye perceives quality (trained on thousands of human ratings). |
| **Deduplication** | **Perceptual Hash (pHash):** Compares visual "fingerprints". | **Embedding Vectors:** Uses AI models (like CLIP) to understand that a "dog" and a "wolf" are similar, even if they look different pixel-wise. |
| **Scale** | **Local Processing:** Runs on one CPU. Good for personal use. | **Distributed Cloud:** Runs on thousands of GPUs simultaneously (e.g., AWS Lambda/Batch). |

**Verdict:** This project represents an **"Industry Standard Lite"** approach. It uses the same fundamental logical steps (Segmentation -> Filtering -> Deduplication) as professional pipelines but swaps out the heavy, expensive AI models for fast, efficient mathematical heuristics. It is perfect for local processing, testing, and understanding the core concepts of Video AI.

---

## 5. Execution Flow: Step-by-Step

Here is the exact journey a video takes when you run the software.

### Step 1: Initialization (`run.py`)
The user types a command in the terminal. `run.py` reads the video file path and sets up the settings (thresholds).

### Step 2: Scene Detection (`src/scene_detector.py`)
The software scans the video looking for "cuts."
*   *Analogy:* Imagine a person clapping their hands every time the camera changes angles. The software marks the time of every clap.
*   *Adaptive Mode:* If requested, the software tries multiple sensitivity levels until it finds the desired number of scenes (e.g., 40).

### Step 3: Extraction (`src/pipeline.py`)
For every "scene" found, the software jumps to the distinct **middle point** of that scene and grabs a single still image.

### Step 4: Quality Gates (`src/quality_control.py`)
Every extracted image must pass a test:
1.  **Sharpness Test:** Is the image too blurry? (Rejected if yes).
2.  **Content Test:** Is it a solid black or white screen? (Rejected if yes).

### Step 5: Deduplication (`src/quality_control.py`)
The software looks at all the "good" images together. If two images look nearly identical (even if they are from different scenes), it keeps only the first one and throws away the copy.

### Step 6: Reporting (`src/visualizer.py`)
Finally, the software saves:
1.  The high-quality JPG images.
2.  A "Contact Sheet" (one big image with all small thumbnails).
3.  An interactive HTML web page report.

---

## 6. Detailed Code Analysis

### A. `run.py` - The Entry Point

**Purpose:** This is the interface for the user. It handles command-line arguments and starts the process.

**Key Functions:**
*   `main()`:
    *   **Input:** Reads command line arguments (Video Path, optional Video ID).
    *   **Action:** Checks if the file exists. Initializes the `KeyframeExtractionPipeline` class with specific settings (e.g., `scene_threshold=27.0`).
    *   **Output:** Triggers the pipeline and prints the final location of the files.

**How to Read:** Look at `if __name__ == "__main__":`. This is where Python starts. It immediately calls `main()`.

---

### B. `src/pipeline.py` - The Manager

**Purpose:** This script acts as the project manager. It doesn't do the heavy lifting itself; it delegates tasks to the specialist scripts.

**Class: `KeyframeExtractionPipeline`**

*   `__init__(...)`:
    *   **Action:** Sets up the folder structure (`data/output/keyframes`). Decides whether to use standard detection or "Adaptive" detection.

*   `process_video(video_path, video_id)`:
    *   **Action:** The "Master Recipe".
        1.  Calls `scene_detector.detect_scenes`.
        2.  Loops through scenes and captures frames (`_extract_keyframes`).
        3.  Loops through frames and runs `quality_control.evaluate_frame`.
        4.  Runs `quality_control.deduplicate_frames`.
        5.  Saves files to disk.
        6.  Saves a JSON report of statistics.
    *   **Output:** A `ProcessingStats` object containing numbers (e.g., "5 scenes found", "2 rejected").

*   `_extract_keyframes(video_path, scenes)`:
    *   **Action:** Opens the video file using OpenCV. Jumps to the specific frame number in the middle of a scene and takes a snapshot.

---

### C. `src/scene_detector.py` - The Eyes

**Purpose:** To analyze video flow and find where scenes change.

**Class: `SceneDetector`**

*   `detect_scenes(video_path)`:
    *   **Logic:** Uses `PySceneDetect`'s `ContentDetector`. It looks at the difference between Frame A and Frame B. If pixels change by more than `threshold` (default 27), it marks a cut.
    *   **Output:** A list of `Scene` objects (start time, end time, frame number).

**Class: `AdaptiveSceneDetector`**

*   `detect_scenes(video_path)`:
    *   **Logic:** This is smarter. It has a "Target" (e.g., "I want 40 scenes").
    *   **Process:**
        1.  It runs detection at a medium threshold.
        2.  If it finds too few scenes, it lowers the threshold (makes it more sensitive).
        3.  If it finds too many scenes, it raises the threshold (makes it less sensitive).
        4.  It repeats this (Binary Search) up to 7 times to get the best result.

---

### D. `src/quality_control.py` - The Gatekeeper

**Purpose:** To use math to judge if an image is "good."

**Class: `QualityControl`**

*   `calculate_blur_score(frame)`:
    *   **Math:** Uses "Laplacian Variance".
    *   **Layman Explanation:** It looks for rapid changes in color (edges).
        *   High Variance = Lots of sharp edges = Crisp Image.
        *   Low Variance = Smooth transitions = Blurry Image.
    *   **Output:** A number (Float). Higher is better.

*   `check_fade_transition(frame)`:
    *   **Logic:** Calculates the average brightness of the image.
    *   **Action:** If the average is near 0 (Black) or near 255 (White) AND the standard deviation is low (the whole image is the same color), it marks it as a "Transition".

*   `deduplicate_frames(frames)`:
    *   **Math:** Uses "Perceptual Hash" (pHash).
    *   **Layman Explanation:** It squashes the image down to a tiny 8x8 grid of simple patterns and turns that into a string of code (e.g., "A1B2...").
    *   **Action:** It compares the code of Frame 1 to Frame 2. If the codes are different by only a few characters (Hamming Distance < 5), it assumes the images are basically the same and deletes the second one.

---

### E. `src/visualizer.py` - The Reporter

**Purpose:** To format the results for human viewing.

**Functions:**

*   `create_contact_sheet(image_paths, output_path)`:
    *   **Action:** Loads all the saved JPGs. Resizes them to small thumbnails. Calculates a grid (rows and columns). Pastes them onto a large black canvas.
    *   **Output:** A single `.jpg` file.

*   `generate_html_report(report_json_path, output_html_path)`:
    *   **Action:** Reads the statistics (processing time, rejection counts). Constructs a standard HTML5 string with CSS styling. Embeds the keyframe images as `<img>` tags.
    *   **Output:** An `.html` file you can open in Chrome or Edge.

---

## 7. Usage Guide

To run this tool on your own machine:

1.  **Activate Environment:** Ensure your Python virtual environment is active.
2.  **Run Command:**
    ```bash
    # Basic usage
    python run.py path/to/video.mp4

    # With a custom name for the output folder
    python run.py path/to/video.mp4 my_vacation_video
    ```
3.  **Find Results:**
    Go to `data/output/keyframes/my_vacation_video/` to see the images.
    Open `data/output/my_vacation_video_report.html` to see the full report.
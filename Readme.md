# Vehicle Counting Solution

## Overview
Classical computer vision solution for counting vehicles in traffic videos captured from a static camera where vehicles move away from the camera.

## Requirements
- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+

## Installation
```bash
pip install -r requirements.txt
```
# Vehicle Detection & Counting – Processing Pipeline

## 1. Background Subtraction (MOG2)
Models each pixel as a **mixture of Gaussians** to separate moving vehicles from the static road background.

## 2. Shadow Removal
Removes shadows by thresholding at pixel value **250**  
(MOG2 marks shadows with value **127**).

## 3. Noise Reduction
Applies:
- **Gaussian Blur** to smooth the foreground mask  
- **Binary Thresholding** to obtain a clean binary image

## 4. Morphological Operations
Applied to refine vehicle shapes:

- **Opening**: Removes small noise particles  
- **Closing**: Fills holes within detected vehicles  
- **Dilation**: Connects fragmented vehicle parts

## 5. Contour Detection & Filtering
Detects **external contours** and filters them based on:

- **Area**  
  - Removes noise (too small)  
  - Removes merged vehicles (too large)

- **Aspect Ratio**  
  - Keeps vehicle-like shapes  
  - Valid range: **0.2 – 5.0**

## 6. Centroid Tracking
Tracks vehicles across frames using **centroid distance matching**:
- Assigns unique IDs
- Maintains identity through temporary occlusions

## 7. Line Crossing Detection
Counts vehicles crossing a **horizontal line at 60% of frame height**:
- Only **upward crossings** are counted
- Represents vehicles moving **away from the camera**

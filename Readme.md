# Vehicle Counting Solution

<a target="_blank" href="https://medium.com/@mohitdeharkar/vehicle-count-80b15739be12">
  <img src="https://medium-widget-api.vercel.app/medium/@mohitdeharkar/vehicle-count-80b15739be12?v=1" alt="Read Vehicle Count on Medium">
</a>

## Overview
Solution for counting vehicles in traffic videos captured from a static camera where vehicles move away from the camera. Counting using classical computer visoin techniques and avoiding deep learning.

| Example 1 | Example 2 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/73efdd71-dced-415a-a589-e3d75dbce2b3" width="400" style="margin-right: 10px;" /> | <img src="https://github.com/user-attachments/assets/89d74b14-3271-4a64-8e21-ce3d4e14df8a" width="400" style="margin-left: 10px;" /> |


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

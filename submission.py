import cv2
import numpy as np
from collections import OrderedDict
import math


class Solution:
    
    def forward(self, video_path):
       
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS) or 25
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        count_line_y = int(height * 0.6)
        
        next_id = 0
        objects = OrderedDict()
        disappeared = OrderedDict()
        counted_ids = set()
        vehicle_count = 0
        
        min_area = int((width * height) / 800)
        max_area = int((width * height) / 8)
        max_distance = int(width * 0.1)
        max_disappeared = int(fps * 1.0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            fg_mask = bg_subtractor.apply(frame)
            fg_mask[fg_mask < 250] = 0
            fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                if aspect < 0.2 or aspect > 5.0:
                    continue
                
                cx = x + w // 2
                cy = y + h // 2
                detections.append((cx, cy, x, y, w, h))
            
            prev_objects = objects.copy()
            
            if len(detections) == 0:
                for oid in list(disappeared.keys()):
                    disappeared[oid] += 1
                    if disappeared[oid] > max_disappeared:
                        del objects[oid]
                        del disappeared[oid]
            
            elif len(objects) == 0:
                for det in detections:
                    objects[next_id] = det
                    disappeared[next_id] = 0
                    next_id += 1
            
            else:
                object_ids = list(objects.keys())
                object_centroids = [(objects[oid][0], objects[oid][1]) for oid in object_ids]
                detection_centroids = [(d[0], d[1]) for d in detections]
                
                D = np.zeros((len(object_centroids), len(detection_centroids)))
                for i, oc in enumerate(object_centroids):
                    for j, dc in enumerate(detection_centroids):
                        D[i, j] = math.sqrt((oc[0] - dc[0])**2 + (oc[1] - dc[1])**2)
                
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_rows, used_cols = set(), set()
                
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    if D[row, col] > max_distance:
                        continue
                    
                    oid = object_ids[row]
                    objects[oid] = detections[col]
                    disappeared[oid] = 0
                    used_rows.add(row)
                    used_cols.add(col)
                
                for row in set(range(len(object_ids))) - used_rows:
                    oid = object_ids[row]
                    disappeared[oid] += 1
                    if disappeared[oid] > max_disappeared:
                        del objects[oid]
                        del disappeared[oid]
                
                for col in set(range(len(detections))) - used_cols:
                    objects[next_id] = detections[col]
                    disappeared[next_id] = 0
                    next_id += 1
            
            for oid in objects:
                if oid in counted_ids:
                    continue
                if oid not in prev_objects:
                    continue
                
                curr_cy = objects[oid][1]
                prev_cy = prev_objects[oid][1]
                
                if prev_cy > count_line_y >= curr_cy:
                    counted_ids.add(oid)
                    vehicle_count += 1
        
        video.release()
        
        return vehicle_count


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    solution = Solution()
    count = solution.forward(video_path)
    print(count)
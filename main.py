import cv2
import numpy as np
import pandas as pd
from tracker import*
from ultralytics import YOLO
import math
import sqlite3

conn = sqlite3.connect('car_counts.db')

# Create a cursor object to interact with the database
cur = conn.cursor()

# Create a table for car counts if it doesn't exist
cur.execute('''
    CREATE TABLE IF NOT EXISTS car_counts (
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        up_count INTEGER,
        down_count INTEGER
    )
''')

# Commit the changes and close the connection
conn.commit()
conn.close()


def save_to_database(up_count, down_count):
    conn = sqlite3.connect('car_counts.db')
    cur = conn.cursor()
    cur.execute('INSERT INTO car_counts (up_count, down_count) VALUES (?, ?)', (up_count, down_count))
    conn.commit()
    conn.close()
# Initialize YOLO model
model = YOLO('yolov10s.pt')

# Class names for YOLO
class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
               9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
               15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 
               23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
               30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
               36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 
               42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
               50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
               58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
               65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
               72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
               79: 'toothbrush'}



class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class CarCount:
    def __init__(self):
        self.count_car = set()

    def test(self, cx, cy, x3, y3, x4, y4, id, resize):
        self.count_car.add(id)
        cv2.circle(resize, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(resize, (x3, y3), (x4, y4), (0, 0, 255), 2)
        return self.count_car

downCar = CarCount()
upCar = CarCount()
tracker = Tracker()
down_count_car = set()
up_count_car = set()

save_down_value = 1
save_up_value = 1

down_area = [(532, 452), (753, 452), (795, 480), (532, 480)]
up_area = [(317, 424), (494, 424), (491, 444), (280, 444)]

cap = cv2.VideoCapture('veh2.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resize = cv2.resize(frame, (980, 720))
    result = model.predict(resize,device = 'cpu')
    a = result[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    ls = []
    for i, r in px.iterrows():
        x1, y1, x2, y2, d = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[5])
        c = class_names[d]
        if 'car' in c:
            ls.append([x1, y1, x2, y2])
    bbox_id = tracker.update(objects_rect=ls)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        down_result = cv2.pointPolygonTest(np.array(down_area, np.int32), (cx, cy), False)
        if down_result >= 0:
            down_count_car = downCar.test(cx, cy, x3, y3, x4, y4, id, resize)
        up_result = cv2.pointPolygonTest(np.array(up_area, np.int32), (cx, cy), False)
        if up_result >= 0:
            up_count_car = upCar.test(cx, cy, x3, y3, x4, y4, id, resize)
    down_num_car = len(down_count_car)
    up_num_car = len(up_count_car)
    
    if down_num_car == save_down_value:
        save_to_database(up_num_car, down_num_car)
        save_down_value+=1
    if up_num_car == save_up_value:
        save_to_database(up_num_car, down_num_car)
        save_up_value+=1
    
    cv2.polylines(resize, [np.array(down_area, np.int32)], True, (255, 255, 255, 0), 3)
    cv2.polylines(resize, [np.array(up_area, np.int32)], True, (255, 255, 255, 0), 3)
    cv2.putText(resize, f"Number of cars going down: {down_num_car}", (60, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 225, 255), 2)
    cv2.putText(resize, f"Number of cars going up: {up_num_car}", (60, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 225, 255), 2)
    cv2.imshow('Video', resize)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


conn = sqlite3.connect('car_counts.db')
cur = conn.cursor()
cur.execute('SELECT * FROM car_counts')
rows = cur.fetchall()
for row in rows:
    print(row)
conn.close()
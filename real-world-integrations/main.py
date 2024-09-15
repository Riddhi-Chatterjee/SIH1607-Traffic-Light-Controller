import os
import random
import math

import cv2
from ultralytics import YOLO

from tracker import Tracker
import time


video_path = os.path.join('.', 'cars1.mp4')
video_out_path = os.path.join('.', 'out1.mp4')
state_path = os.path.join('.', 'state_space.txt')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")
model.to('cuda')

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
t1 = time.time()
fc=0

old_position = {}
new_position = {}

old_velocities = {}
new_velocities = {}

while ret:

    if(fc == 62): break


    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 2:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            new_position[track_id] = [(x1+x2)/2, (y1+y2)/2]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    if(fc%6 == 0):
        # print(old_position)
        # print(new_position)
        velocities = []
        accelerations = []
        rest_count = 0
        for t_id in new_position:
            if t_id in old_position:
                x1, y1 = old_position[t_id]
                x2, y2 = new_position[t_id]
                d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                v = d/(0.2)
                if(v!=0):velocities.append(str(v))
                rest_count +=1
                new_velocities[t_id] = v

        for t_id in new_velocities:
            if t_id in old_velocities:
                v1 = old_velocities[t_id]
                v2 = new_velocities[t_id]
                a = (v2-v1)/(0.2)
                accelerations.append(str(a))
        
        old_position = new_position.copy()
        old_velocities = new_velocities.copy()

        new_position = {}
        new_velocities = {}
        
        with open(state_path, 'a') as file:
            file.write(f'After {fc} frames')
            file.write('\n')
            text = ' '.join(velocities)
            if(fc!=0): file.write(text)
            file.write('\n')
            if(fc!=0): file.write(str(rest_count))
            file.write('\n')
            text = ' '.join(accelerations)
            if(fc>6): file.write(text)
            
            file.write('\n\n')
            
    fc+=1

    cap_out.write(frame)
    ret, frame = cap.read()


cap.release()
cap_out.release()
cv2.destroyAllWindows()
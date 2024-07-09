import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
import time
import requests  # Import the requests library

# Load the pre-trained YOLOv8 model for person detection
model = YOLO('yolov8s.pt')

# Define the vertical area for detecting passing people (spans full height of the screen)
frame_width = 1020
vertical_area = [(frame_width // 3 - 50, 0), (frame_width // 3 + 50, 0), 
                 (frame_width // 3 + 50, 500), (frame_width // 3 - 50, 500)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize the video capture
cap = cv2.VideoCapture("peoplecount1.mp4")

# Read class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

people_detected = {}
detected = set()

class Tracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.last_detected = {}
        self.maxDisappeared = maxDisappeared
        self.disappeared_time_threshold = 2  # seconds

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.last_detected[self.nextObjectID] = time.time()
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.last_detected[objectID]

    def update(self, rects):
        if len(rects) == 0:
            current_time = time.time()
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if (current_time - self.last_detected[objectID]) > self.disappeared_time_threshold:
                    self.deregister(objectID)
            return [(objectID, *self.objects[objectID]) for objectID in self.objects]

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.last_detected[objectID] = time.time()
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if (time.time() - self.last_detected[objectID]) > self.disappeared_time_threshold:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return [(objectID, *self.objects[objectID]) for objectID in self.objects]

tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, 500))

    # Detect people using YOLOv8
    results = model(frame)
    rects = []
    if results is not None and len(results) > 0:
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # YOLO class 0 is for person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    rects.append((x1, y1, x2, y2))

    bbox_id = tracker.update(rects)

    for bbox in bbox_id:
        id, cx, cy = bbox

        # Using centroid (cx, cy) and ensuring it's in tuple format with integer values
        point = (int(cx), int(cy))

        results = cv2.pointPolygonTest(np.array(vertical_area, np.int32), point, False)
        if results >= 0:
            if id not in people_detected:  # Only trigger for new detections
                # Send POST request to the endpoint with the key
                payload = {
                    'key': '47MzVlvfaUdNUDiOFlvswysdgrxW8aPeUW2oLF1NW1C92Ibfbe',
                }
                requests.post("https://people-count.smaster.live/api.php", data=payload)

            people_detected[id] = point
            cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
            cv2.circle(frame, point, 4, (255, 0, 255), -1)
            cv2.putText(frame, str(id), (cx + 15, cy - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            detected.add(id)

    # Draw the vertical area on the frame
    cv2.polylines(frame, [np.array(vertical_area, np.int32)], True, (255, 0, 0), 2)

    # Display detected count
    count = len(detected)
    cv2.putText(frame, 'Number of detected people= ' + str(count), (20, 44), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

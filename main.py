import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
from deep_sort.tracker import Tracker

# Set video source
video_path = os.path.join(".", "Img", "test_01.mp4")
print(video_path)

# Create a video capture object and read from input file
video_cap = cv2.VideoCapture(video_path)

# Export model
model = YOLO("yolov8s.pt")

# Initialize Track object
people_tracker = Tracker()

# Define bounding boxes coloros
bbox_colors = []

for i in range(25):

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    bbox_colors.append([r, g, b])

# Read video
if video_cap.isOpened():

    while video_cap.isOpened():

        # Set and show new video frame
        ret, frame = video_cap.read()
        cv2.imshow("Output", frame)

        # Predict results per frame
        results = model.predict(frame, classes=[0], device="cpu")

        # Check all results for prediction
        for result in results:

            person_bboxes = []

            # Get bounding boxes parameters
            for data in result.boxes.data.tolist():

                x1, y1, x2, y2, score, class_id = data

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                class_id = int(class_id)

                # Save bounding boxes parameters
                person_bboxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        score,
                    ]
                )

                # Update bounding boxes parameters per frame
                people_tracker.update(frame, person_bboxes)

                # Track bounding boxes
                for track in people_tracker.tracks:

                    bbox = track.bbox
                    track_id = track.track_id

                    # Display bounding boxes
                    bbox_size = 2
                    x1y1 = (int(bbox[0]), int(bbox[1]))
                    x2y2 = (int(bbox[2]), int(bbox[3]))
                    color = bbox_colors[random.randint(0, 9)]

                    cv2.rectangle(frame, x1y1, x2y2, color, bbox_size)

        # Press 'q' on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

# Closes all frames
video_cap.release()
cv2.destroyAllWindows()

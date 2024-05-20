import os
import cv2
import numpy as np
import tensorflow as tf
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools.generate_detections import create_box_encoder
from deep_sort.application_util import visualization

if __name__ == "__main__":

    # Paths definitions
    video_path = os.path.join(".", "Img", "test_01.mp4")
    model_path = os.path.join(".", "resources", "networks, ", "mars-small128")
    output_path = os.path.join(".", "output_video.mp4")

    # Initialize Deep SORT components
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    encoder = create_box_encoder(model_path, batch_size=32)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )

    people_tracker = Tracker(metric)

    # Open video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():

        print("Error opening video file")

    else:

        # Prepare video writer

        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"MP4"),
            30.0,
            (frame_width, frame_height),
        )

# Real-Time Person Tracking

## Overview

This project aims to implement real-time tracking of individuals throughout a video using advanced computer vision techniques. It leverages the Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) algorithm for robust and efficient tracking. The project uses the [Deep SORT repository](https://github.com/nwojke/deep_sort) as a foundational base to track and identify people across frames in a video stream.

## Key Features

- **Real-Time Tracking:** Provides accurate and efficient tracking of multiple individuals in video footage.
- **Integration with Deep SORT:** Utilizes the Deep SORT algorithm to enhance tracking accuracy and maintain identity consistency.
- **Scalable Solution:** Designed to handle various video resolutions and frame rates.

## Hardware and Software Requirements

### Hardware

- A computer with a capable GPU for real-time processing (optional but recommended for improved performance).

### Software

- **Python 3.x**: For running the tracking scripts.
- **OpenCV**: For video processing and handling.
- **Deep SORT**: For tracking implementation.
- **TensorFlow/PyTorch**: Depending on the version of Deep SORT used, the model may require TensorFlow or PyTorch.

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nwojke/deep_sort.git

2. **Install Dependencies:**

   Navigate to the cloned repository directory and install the required Python libraries:
   ```bash
   cd deep_sort
   pip install -r requirements.txt

3. **Prepare Your Video:**

    Place the video file you want to process in the project directory or specify its path in the script.


4. **Run the Tracking Script:**

    Execute the tracking script to start processing the video and obtain real-time tracking results:

    ```bash
    Copy code
    python track.py --video your_video_file.mp4

### Scripts Description

- **`track.py`**: Main script for running the tracking algorithm. It initializes the Deep SORT model and processes the video frame by frame to track individuals.
- **`deep_sort/`**: Contains the Deep SORT implementation, including the tracking algorithm and associated metrics for identity association.

### Usage

- **Play/Pause Video Tracking:** Control video playback while tracking to analyze different sections.
- **Export Tracking Data:** Save tracking results, including bounding boxes and tracking IDs, for further analysis.

### Contributions

Contributions to enhance the tracking system or extend its capabilities are welcome. Please fork the repository and submit pull requests for review.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements

- Special thanks to the developers of the Deep SORT algorithm for providing a robust tracking framework.

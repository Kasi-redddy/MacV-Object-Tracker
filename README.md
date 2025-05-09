# MacV Object Tracker

A Flask-based web application for object tracking in videos using YOLOv8, with visualizations, metrics, and browser-based results.

---

## About

**MacV Object Tracker** is a web application that tracks objects in a video, visualizes their movement, and computes key metrics such as total time spent and number of unique objects. The results are displayed in a browser with an interactive video player and downloadable output.

---

## Features

- **Object Tracking:** Detects and tracks objects across video frames using YOLOv8.
- **Metrics Calculation:**
  - Total time each object spends in the video
  - Total number of unique object IDs detected
- **Visualization:**
  - Bounding boxes and centroids for each object
  - Trail lines showing object movement
- **Web Interface:** Simple Flask frontend for processing and viewing results
- **Export:** Processed video is browser-playable and downloadable
- **Documentation:** Clear instructions and code comments

---

## Tech Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Frontend:** HTML5, Jinja2 (Flask templates)
- **Other:** Numpy

---

### Prerequisites

pip install -r requirements.txt
 
---

## Approach & Implementation

- **Detection & Tracking:** Uses YOLOv8 for robust object detection and tracking.
- **Metrics:** Tracks unique object IDs and calculates time spent in the frame.
- **Visualization:** Draws bounding boxes, centroids, and trails for each object.
- **Frontend:** Flask serves HTML5 video and metrics, with a download link for the output.
- **Export:** Uses OpenCV’s `mp4v` codec for browser compatibility.

**Key Decisions:**
- Chose YOLOv8 for state-of-the-art accuracy and built-in tracking.
- Used OpenCV for video processing and drawing.
- Used Flask for a simple, effective web interface.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---


## Contact

For support or questions, please contact [reddykasivisweswar@gmail.com].


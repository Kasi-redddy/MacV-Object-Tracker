import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP workaround for Windows

import cv2
import numpy as np
from flask import Flask, render_template, send_file
from ultralytics import YOLO
from collections import defaultdict
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_videos'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load YOLO model (downloads weights if not present)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt for better accuracy

def process_video(input_path):
    track_history = defaultdict(list)
    object_times = defaultdict(list)
    unique_ids = set()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' (H.264) for browser compatibility
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("VideoWriter failed to open! Trying fallback codec.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("VideoWriter could not be opened with any codec!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO tracking inference
        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                unique_ids.add(track_id)
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                object_times[track_id].append(current_time)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                centroid = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

                track_history[track_id].append(centroid)
                history = track_history[track_id][-30:]  # Last 30 points
                for i in range(1, len(history)):
                    cv2.line(frame, history[i-1], history[i], (255, 0, 0), 2)

        out.write(frame)

    # Metrics calculation
    metrics = {
        'total_objects': len(unique_ids),
        'object_durations': {
            tid: max(times) - min(times) if len(times) > 1 else 0
            for tid, times in object_times.items()
        }
    }

    cap.release()
    out.release()
    return output_path, metrics

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process')
def process():
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    if not os.path.exists(video_path):
        return "Video file not found. Please place 'input_video.mp4' in the 'uploads' folder."
    start_time = time.time()
    output_path, metrics = process_video(video_path)
    processing_time = time.time() - start_time
    return render_template('result.html', metrics=metrics, processing_time=processing_time)

@app.route('/video')
def video():
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output.mp4')
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)

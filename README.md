# Multimodal Object Detector 
Real-Time Object Detection & Color Analysis using YOLO and OpenCV
📌 Overview

This project is a real-time computer vision system that performs object detection on live webcam feed using a YOLO-based model and provides additional analytics such as:
- Object counting by class
- Class-based filtering (person / vehicles / other objects)
- Dominant color estimation of detected objects
- FPS and detection-time monitoring
- Screenshot capture
- CSV logging of detections

The system is designed to be interactive, efficient, and suitable for basic surveillance or scene analysis applications.

Project Structure
.
├── detector.py          # YOLO model loading and detection functions
├── webcam.py            # Camera initialization and frame reading
├── clustercheck.py      # Color clustering and closest color matching
├── main.py              # Main application loop (this script)
├── hello.csv            # Detection log file (auto-generated)
└── README.md

CSV Logging Format
When logging is enabled, detections are saved in hello.csv with the following columns:

timestamp, class_name, confidence, xmin, ymin, xmax, ymax

This allows for later analysis such as: Object frequency, Activity over time, Spatial heatmaps

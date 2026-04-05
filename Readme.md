# 🧠 Computer Vision Project — Child Tracking & Group Detection

## 📌 Project Overview
This project focuses on building a Computer Vision system that detects and tracks individuals (adults and children) in video streams and analyzes their relationships (grouping behavior).

The system uses deep learning (YOLO) for detection and tracking algorithms to maintain identity across frames, enabling real-time monitoring and event detection.

---

## 👥 Team Information

Name: [shamah Abdullah]  
Role:  ML Engineer  
Contribution: Model training, system design, pipeline integration  

Name: [Alaa Monshi]  
Role: Team Leader / Data Engineer  
Contribution: Dataset collection, preprocessing, labeling  

Name: [Fawaz Mufti]  
Role: Computer Vision Engineer / Developer  
Contribution: Tracking, grouping logic, implementation, testing  

---

## 🎯 Project Objectives

- Detect adults and children in video streams  
- Track individuals across frames using object tracking  
- Identify groups (child + adult) based on:
  - Distance
  - Time window  
- Generate alerts when:
  - A child is alone  
  - A child leaves a group  

### Why This Problem Matters
- Useful for child safety systems (malls, public spaces)  
- Enables smart surveillance  
- Reduces manual monitoring effort  

### Expected Outcome
- Real-time detection and tracking system  
- Group behavior analysis  
- Alert generation system  

---

## 📂 Dataset

- Source: Custom dataset + Roboflow  
- Type: Images and extracted video frames  
- Classes:
  - Child  
  - Adult  

### Dataset Details
- ~2000+ frames  
- Multiple camera angles and environments  

### Preprocessing Steps
- Image resizing  
- Annotation (manual + auto-labeling via Roboflow)  
- Data cleaning (removing incorrect labels)  
- Frame extraction from videos  

---

## 🧠 Methodology

### 1. Data Preprocessing
- Extract frames from videos  
- Label adults and children  
- Improve dataset diversity (angles, occlusions)  

### 2. Model Selection
- YOLOv8s for object detection  
- ByteTrack / DeepSORT for tracking  

### 3. Pipeline
Detection (YOLO)  
↓  
Tracking (ByteTrack)  
↓  
Group Manager  
↓  
Event Detection  
↓  
Alert System  

### 4. Group Logic
- Distance threshold between people  
- Time window (e.g., 4 seconds)  
- Assign Group ID if:
  - Child is close to adult for a period  

### 5. Evaluation
- Detection Accuracy (~93%)  
- Visual inspection of tracking  
- Event correctness (group vs alone)  

---

## ⚙️ Implementation

### Tools and Libraries
- Python  
- OpenCV  
- Ultralytics YOLOv8  
- NumPy  
- PyTorch  

### Key Components
- main.py → Runs the full pipeline  
- group_manager.py → Handles grouping logic  
- alert_engine.py → Generates alerts  
- tracker.py → Tracks individuals  

### Challenges Faced

1. Misclassification (far people detected as child)  
   - Cause: limited dataset diversity  
   - Solution: added more varied frames  

2. Occlusion (half body visible)  
   - Solution: improved dataset with partial views  

3. Large labeling effort  
   - Solution: used auto-labeling (Roboflow)  

4. Identity tracking issues  
   - Solution: integrated tracking (ByteTrack)  

---

## 📊 Results

### Detection Performance
- Accuracy: ~93%  
- Good performance in normal conditions  

### Limitations
- Errors when:
  - Person is far from camera  
  - Body partially hidden  

### Example Output
EVENT: Child grouped with Adult (Group ID: 2)  
ALERT: Child alone detected  

---

## 🚀 How to Run the Project

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run the System
python main.py

### 3. Notes
- Make sure model weights are in:
  /yolo/best.pt  
- Update video path inside main.py if needed  

---

## 🧠 Summary

This project demonstrates:
- Object detection using YOLO  
- Multi-object tracking  
- Behavior analysis (group detection)  
- Real-time alert generation  

A complete end-to-end Computer Vision system suitable for real-world applications.
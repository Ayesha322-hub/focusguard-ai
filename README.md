# 🛡️ FocusGuard AI

**Unified Attention Monitoring System** for Drivers and Students using Computer Vision.

## ✨ Features

### 🚗 Driver Mode
- Drowsiness Detection (Eye Aspect Ratio)
- Yawn Detection (Mouth Aspect Ratio)
- Phone Usage Detection (YOLOv8)
- Head Pose Tracking (Eyes off road)
- Object Distraction (food, drinks)
- Real-time audio alerts
- Safety Score with grade (A-F)

### 📚 Student Mode
- All driver features (study tools allowed)
- Absence Detection (left seat)
- Multi-Face Detection (cheating prevention)
- Focus % Time Tracking
- Focus Score with grade (A+ to D)

## 🛠️ Tech Stack

- **MediaPipe** - Face mesh & landmark detection
- **YOLOv8 (Ultralytics)** - Object detection (phone, food, etc.)
- **OpenCV** - Image processing & camera handling
- **Streamlit** - Web UI (mobile + desktop friendly)
- **NumPy** - Numerical computations

## 🚀 Quick Start

\\\ash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/focusguard-ai.git
cd focusguard-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
\\\

Open browser → http://localhost:8501

## 📱 Mobile Access

When running locally, the app is accessible from any device on the same WiFi:
- Open \http://YOUR_LAPTOP_IP:8501\ on phone browser
- Same UI works on mobile

## 📊 Reports

Each session generates a JSON report saved to \eports/\ with:
- Safety/Focus Score
- Time breakdown (focused/distracted/absent)
- Event log with timestamps
- Per-detector statistics

## 🏗️ Project Structure

\\\
focusguard-ai/
├── app.py                          # Streamlit web app
├── requirements.txt
├── focusguard/
│   ├── core/
│   │   └── vision_engine.py        # MediaPipe wrapper
│   ├── models/
│   │   ├── eye_state.py            # EAR drowsiness
│   │   ├── yawn_detector.py        # MAR yawn + alarm
│   │   ├── phone_detector.py       # YOLO phone
│   │   ├── head_pose.py            # 3D head pose
│   │   └── object_distraction.py   # YOLO objects
│   └── modes/
│       ├── driver_mode.py          # Standalone CLI
│       └── student_mode.py         # Standalone CLI
└── reports/                         # Auto-generated
\\\

## 📝 License

MIT

## 👨‍💻 Author
Amara Tariq 

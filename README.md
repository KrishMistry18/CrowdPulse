<div align="center">

# 🎥 CrowdPulse

### Real-time Crowd Analysis Dashboard

[![Live Demo](https://img.shields.io/badge/🔗%20Live%20Demo-Live%20App-2D6A4F?style=for-the-badge)](https://huggingface.co/spaces/KrishMistry18/CrowdPulse)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*A full-stack web application built with Flask and YOLOv8 to analyze crowd behavior in real-time. This system detects, tracks, and analyzes crowd density, speed, and directional chaos to provide early warnings for potential stampedes.*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔒 **Secure Web Dashboard** | A Flask-based web app with a passkey-protected login |
| 🎯 **Dynamic "Hotspot" Zones** | Upload any video and define custom regions of interest |
| 📊 **Advanced Behavioral Analysis** | Tracks crowd density, average speed, and directional chaos |
| 🧠 **Self-Calibrating Model** | Automatically learns a "normal" baseline for any video and flags anomalies |
| 🚨 **Real-time Alerts** | Classifies the crowd state as `NORMAL`, `WARNING`, or `CRITICAL` |
| 🚀 **GPU Accelerated** | Uses PyTorch with CUDA to perform all AI inference on an NVIDIA GPU |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask |
| Computer Vision | OpenCV, YOLOv8 (`yolov8m`), `supervision` |
| AI / Deep Learning | PyTorch (CUDA 12.1) |
| Frontend | HTML5, CSS3, Jinja2 |
| Deployment | Docker (Hugging Face Spaces) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- NVIDIA GPU (Recommended for real-time inference)
- CUDA 12.1 toolkit

### 1. Clone the Repository

```bash
git clone https://github.com/KrishMistry18/CrowdPulse.git
cd CrowdPulse
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install PyTorch (GPU)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Download the AI Model

- Download the **`yolov8m.pt`** model from the [YOLOv8 releases](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt).
- Place the `.pt` file in the main `CrowdPulse` folder.

### 5. Run the Server

```bash
python app.py
```

Open **http://127.0.0.1:5000** 🎉

---

## 📁 Project Structure

```
CrowdPulse/
├── app.py                  # Core Flask server and routes
├── main_web.py             # Main inference & video processing logic
├── requirements.txt        # Dependencies
├── Dockerfile              # Hugging Face deployment config
├── yolov8m.pt              # YOLOv8 weights (downloaded)
├── static/                 # CSS, JS, and images
├── templates/              # Jinja2 HTML templates
└── uploads/                # Directory for uploaded videos
```

---

## 🧠 Model Architecture

CrowdPulse combines several cutting-edge computer vision techniques:

| Component | Function |
|---|---|
| **YOLOv8 Medium** | Real-time object detection (optimized for 'person' class) |
| **Supervision** | Multi-object tracking, zone detection, and annotation rendering |
| **Motion Tracking** | Used to calculate movement vectors and directional chaos |
| **Anomaly Detection** | Self-calibrating statistical analysis to detect panic states |

---

## 🔒 Security & Performance

- **Passkey Protection** — Limits access to the analytics dashboard.
- **Async Processing** — Video processing runs in background threads to keep the UI responsive.
- **Auto-Cleanup** — Temporary video files are automatically managed.
- **GPU Optimization** — Operations are tensorized and offloaded to CUDA where possible.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request

---

## License

MIT — see `LICENSE` for details.

---

<div align="center">

*Made for safety · CrowdPulse v1.0.0 · Built by [Krish Mistry](https://github.com/KrishMistry18)*

</div>
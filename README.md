# Crowd-Pulse: Real-time Crowd Analysis Dashboard

A full-stack web application built with Flask and YOLOv8 to analyze crowd behavior in real-time. This system detects, tracks, and analyzes crowd density, speed, and directional chaos to provide early warnings for potential stampedes.

![Screenshot of Dashboard](link-to-a-screenshot-of-your-dashboard.png)
*(Tip: Take a screenshot of your dashboard, add it to your folder, and change the link)*

## ✨ Core Features
* **Secure Web Dashboard:** A Flask-based web app with a passkey-protected login.
* **Dynamic "Hotspot" Zones:** Users can upload any video and define custom regions of interest.
* **Advanced Behavioral Analysis:** Tracks three key metrics:
    1.  **Crowd Density:** Counts people in total and within hotspots.
    2.  **Average Crowd Speed:** Calculates the average speed of all individuals.
    3.  **Directional Chaos:** Measures the variance of motion angles to detect panic.
* **Self-Calibrating Model:** Automatically learns a "normal" baseline for any video and flags statistical anomalies.
* **Real-time Alerts:** Classifies the crowd state as `NORMAL`, `WARNING`, or `CRITICAL` and displays it on the stream.
* **GPU Accelerated:** Uses PyTorch with CUDA to perform all AI inference on an NVIDIA GPU.

## 🛠️ Tech Stack
* **Backend:** Python 3.9, Flask
* **Computer Vision:** OpenCV, YOLOv8 (`yolov8m`), `supervision`
* **GPU:** PyTorch (CUDA 12.1)
* **Frontend:** HTML5, CSS3, Jinja2

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Python libraries:**
    ```bash
    pip install -r requirements.txt 
    ```
    *(Note: You need to create this file by running `pip freeze > requirements.txt`)*

4.  **Install PyTorch (GPU):**
    ```bash
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
5.  **Download the AI Model:**
    * Download the **`yolov8m.pt`** model from the [YOLOv8 releases](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt).
    * Place the `.pt` file in the main `crowd_pulse_web` folder.

6.  **Run the server:**
    ```bash
    python app.py
    ```
7.  Open your browser and go to `http://127.0.0.1:5000`.
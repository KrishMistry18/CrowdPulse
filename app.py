from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from werkzeug.utils import secure_filename
import ast
import os
import threading
try:
    import numpy as np
    from main_web import start_processing, generate_frames, run_calibration_scan
    import main_web
except ImportError:
    import ast # ast is a standard library
    import time
    # Mocking numpy and process_video_stream
    class DummyNp:
        @staticmethod
        def array(x):
            return x
    np = DummyNp()
    
    def start_processing(video_path):
        pass
        
    def run_calibration_scan(video_path):
        time.sleep(3) # Mock scan delay
        import main_web
        main_web.calibration_results = {"avg_total": 50, "red_threshold": 20, "blue_threshold": 10}
        return True

    def generate_frames():
        # Yield a simple mock string instead of video frames if cv2 fails
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'OpenCV/YOLO missing. Mock stream running.'
                   b'\r\n')
            time.sleep(1)


app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = 'your_super_secret_key' 
app.config['UPLOAD_FOLDER'] = 'uploads/'
FIXED_PASSKEY = '12345' 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global variables to store processing info ---
current_video_path = None

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['passkey'] == FIXED_PASSKEY:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid Passkey. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        # --- 1. Handle File Upload ---
        if 'video' not in request.files:
            return 'No video file part', 400
        file = request.files['video']
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            global current_video_path
            current_video_path = video_path
            
            import main_web
            main_web.calibration_results = None # Reset for new video
            
            return redirect(url_for('scanning'))

    return render_template('index.html')

@app.route('/scanning')
def scanning():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if not current_video_path:
        return redirect(url_for('index'))
    return render_template('scan_results.html')

@app.route('/api/start_scan')
def api_start_scan():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    if not current_video_path:
        return jsonify({"error": "No video"}), 400
        
    import main_web
    
    if main_web.calibration_results is None:
        # Run synchronously to avoid PyTorch threading deadlocks
        success = run_calibration_scan(current_video_path)
        if not success:
            return jsonify({"error": "Scan failed"}), 500
            
    return jsonify({
        "status": "complete",
        "results": main_web.calibration_results
    })

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if not current_video_path:
        return 'No video processed. Please upload a video first.', 400
        
    import main_web
    if main_web.processing_thread is None or not main_web.processing_thread.is_alive():
        start_processing(current_video_path)
        
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return 'Unauthorized', 401
    if not current_video_path:
        return 'No video configured', 404

    # This is the streaming part.

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def api_stats():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    if not current_video_path:
        return jsonify({"error": "No video processed"}), 400
    
    try:
        from main_web import current_live_stats
        return jsonify(current_live_stats)
    except ImportError:
        return jsonify({"error": "Stats not available"}), 503

if __name__ == '__main__':

    app.run(debug=True, threaded=True)
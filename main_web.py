import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import math
import threading
import time
import scipy.cluster.hierarchy as hcluster

# --- Configuration ---
MODEL_NAME = 'yolov8m.pt' 
CALIBRATION_FRAMES = 150 

current_live_stats = {
    "total_count": 0,
    "avg_speed": 0.0,
    "chaos_metric": 0.0,
    "status_text": "WAITING",
    "is_warning": False,
    "is_critical": False
}


current_frame = None
processing_thread = None
stop_event = threading.Event()
calibration_results = None
stats_history = []

print("Loading YOLO model globally...")
yolo_model = YOLO(MODEL_NAME)
print("YOLO model loaded successfully!")

def run_calibration_scan(video_path):
    global calibration_results
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return False

    model = yolo_model
    
    total_counts = []
    
    print("--- Starting Background Calibration Scan ---")
        
    frame_count = 0
    while cap.isOpened() and frame_count < CALIBRATION_FRAMES:
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame, classes=0, verbose=False, conf=0.25)
        detections = sv.Detections.from_ultralytics(results[0])
        total_counts.append(len(detections))
        frame_count += 1
        
    cap.release()
    
    if not total_counts:
        avg_total = 0
    else:
        avg_total = np.mean(total_counts)
        
    # Calculate Dynamic Thresholds based on average total crowd size
    # Fallback to hardcoded absolute minimums so it doesn't break on sparse videos
    red_threshold = max(10, int(avg_total * 0.40)) # 40% of average crowd
    blue_threshold = max(5, int(avg_total * 0.15)) # 15% of average crowd
    
    calibration_results = {
        "avg_total": float(round(avg_total, 1)),
        "red_threshold": int(red_threshold),
        "blue_threshold": int(blue_threshold)
    }
    
    print(f"--- Calibration Scan Complete: {calibration_results} ---")
    return True

def start_processing(video_path):
    global processing_thread, stop_event, current_frame, current_live_stats, stats_history
    if processing_thread is not None and processing_thread.is_alive():
        stop_event.set()
        processing_thread.join()
        
    stop_event.clear()
    current_frame = None
    
    # Reset stats
    current_live_stats.update({
        "total_count": 0,
        "avg_speed": 0.0,
        "chaos_metric": 0.0,
        "status_text": "WAITING",
        "is_warning": False,
        "is_critical": False,
        "progress": 0
    })
    stats_history.clear()

    processing_thread = threading.Thread(target=_process_video_stream_worker, args=(video_path,))
    processing_thread.daemon = True
    processing_thread.start()

def generate_frames():
    global current_frame
    while True:
        if current_frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               current_frame + b'\r\n')
        time.sleep(0.03)

def _process_video_stream_worker(video_path):
    global current_live_stats, current_frame, stop_event
    
    # Initialize a loading frame so the browser doesn't wait and time out
    loading_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(loading_img, "LOADING AI MODEL... PLEASE WAIT", (400, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    flag, encoded = cv2.imencode(".jpg", loading_img)
    if flag:
        current_frame = bytearray(encoded)
        
    current_live_stats["status_text"] = "LOADING AI MODEL..."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
    
    model = yolo_model
    tracker = sv.ByteTrack()

    # Dynamic clusters instead of static zones
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_position=sv.Position.TOP_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=100)
    
    previous_positions = defaultdict(lambda: None)
    
    calibration_data = {
        "avg_speeds": [], "total_counts": [],
        "chaos_metrics": [] 
    }
    baseline_stats = {}
    frame_count = 0

    current_live_stats["status_text"] = "PROCESSING FIRST FRAME..."
    print("--- Starting Video Stream Processing ---")

    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            print("End of video file.")
            break

        # --- Run Inference ---
        results = model(frame, classes=0, verbose=False, conf=0.25)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked_detections = tracker.update_with_detections(detections)

        # --- Calculations (Speed, Chaos, Count) ---
        current_speeds = []
        current_angles = [] 
        
        if tracked_detections.tracker_id is not None:
            current_centroids = tracked_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            for tracker_id, centroid in zip(tracked_detections.tracker_id, current_centroids):
                prev_pos = previous_positions[tracker_id]
                if prev_pos is not None:
                    distance = math.sqrt((centroid[0] - prev_pos[0])**2 + (centroid[1] - prev_pos[1])**2)
                    current_speeds.append(distance)
                    if distance > 1: 
                        dx = centroid[0] - prev_pos[0]
                        dy = centroid[1] - prev_pos[1]
                        angle = math.atan2(dy, dx)
                        current_angles.append(angle)
                previous_positions[tracker_id] = centroid

        avg_speed = sum(current_speeds) / len(current_speeds) if current_speeds else 0.0
        chaos_metric = np.std(current_angles) if len(current_angles) > 1 else 0.0
        total_count = len(tracked_detections)
        
        current_centroids_list = []
        if tracked_detections.tracker_id is not None:
            current_centroids_list = tracked_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            
        # --- Dynamic Clustering ---
        clusters = {}
        if len(current_centroids_list) > 1:
            # Cluster centroids within 200 pixels of each other
            cluster_ids = hcluster.fclusterdata(current_centroids_list, 200, criterion="distance")
            for pt, cid in zip(current_centroids_list, cluster_ids):
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(pt)
        elif len(current_centroids_list) == 1:
            clusters[1] = [current_centroids_list[0]]

        annotated_frame = frame.copy() 

        # --- Prediction Phase ---
        status_text = "NORMAL"
        status_color = (0, 255, 0) # Green
        is_warning = False
        is_critical = False
        
        red_zones = 0
        blue_zones = 0
        green_zones = 0
        
        # We will determine cluster colors first, then set stats
        red_threshold = calibration_results["red_threshold"] if calibration_results else 10
        blue_threshold = calibration_results["blue_threshold"] if calibration_results else 5
        
        # --- Annotation ---
        # Draw Dynamic Cluster Zones
        for cid, pts in clusters.items():
            cluster_count = len(pts)
            pts_arr = np.array(pts)
            min_x, min_y = np.min(pts_arr, axis=0) - 40
            max_x, max_y = np.max(pts_arr, axis=0) + 40
            
            # Bound within frame
            min_x = max(0, int(min_x))
            min_y = max(0, int(min_y))
            max_x = min(frame_width, int(max_x))
            max_y = min(frame_height, int(max_y))
            
            if cluster_count >= red_threshold:
                zone_color = (0, 0, 255) # Red
                red_zones += 1
            elif cluster_count >= blue_threshold:
                zone_color = (255, 0, 0) # Blue
                blue_zones += 1
            else:
                zone_color = (0, 255, 0) # Green
                green_zones += 1
                    
            # Draw the dynamic bounding box
            cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), zone_color, 4)
            
            # Draw the label
            label_text = f"Group: {cluster_count}"
            cv2.putText(annotated_frame, label_text, (min_x, max(30, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
            cv2.putText(annotated_frame, label_text, (min_x, max(30, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 2)

        if red_zones > 0:
            is_warning = True
            status_text = "WARNING (CLUSTER OVERCROWDED)"
            status_color = (0, 255, 255)
        elif calibration_results and total_count > calibration_results["avg_total"] * 1.5:
            is_warning = True
            status_text = "WARNING (HIGH OVERALL DENSITY)"
            status_color = (0, 255, 255)
            
        current_live_stats["total_count"] = total_count
        current_live_stats["avg_speed"] = avg_speed
        current_live_stats["chaos_metric"] = chaos_metric
        current_live_stats["status_text"] = status_text
        current_live_stats["is_warning"] = is_warning if 'is_warning' in locals() else False
        current_live_stats["is_critical"] = is_critical if 'is_critical' in locals() else False
        
        progress = int((frame_count / total_frames) * 100)
        current_live_stats["progress"] = min(99, progress) # Keep at 99 until truly done

        stats_history.append({
            "time": time.time(),
            "frame": frame_count,
            "total_count": total_count,
            "avg_speed": round(avg_speed, 2),
            "chaos_metric": round(chaos_metric, 2),
            "red_zones": red_zones,
            "blue_zones": blue_zones,
            "green_zones": green_zones,
            "status": status_text
        })
        frame_count += 1

        # (The cluster drawing was moved above)

        # Draw Detections
        labels = [f"#{tracker_id}" for tracker_id in tracked_detections.tracker_id] if tracked_detections.tracker_id is not None else []
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        # Draw Stats
        cv2.putText(annotated_frame, f"Total Count: {total_count}", (frame_width - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(annotated_frame, f"Total Count: {total_count}", (frame_width - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(annotated_frame, f"Avg Speed: {avg_speed:.2f}", (frame_width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(annotated_frame, f"Avg Speed: {avg_speed:.2f}", (frame_width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(annotated_frame, f"Chaos: {chaos_metric:.2f}", (frame_width - 400, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3) 
        cv2.putText(annotated_frame, f"Chaos: {chaos_metric:.2f}", (frame_width - 400, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) 
        
        # Draw Final Status
        cv2.putText(annotated_frame, status_text, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 7, cv2.LINE_AA)
        cv2.putText(annotated_frame, status_text, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 4, cv2.LINE_AA)

        frame_count += 1 

        # Encode the frame as a JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if flag:
            current_frame = bytearray(encodedImage)

    cap.release()
    
    # Send final "PROCESSING COMPLETED" frame
    if annotated_frame is not None:
        cv2.putText(annotated_frame, "PROCESSING COMPLETED", (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 10, cv2.LINE_AA)
        cv2.putText(annotated_frame, "PROCESSING COMPLETED", (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
        
        current_live_stats["status_text"] = "PROCESSING COMPLETED"
        current_live_stats["is_warning"] = False
        current_live_stats["is_critical"] = False
        current_live_stats["progress"] = 100
        
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if flag:
            current_frame = bytearray(encodedImage)
            
    print("--- Video Stream Processing Finished ---")
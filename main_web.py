import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import math
import threading
import time

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

def start_processing(video_path, zone_polygons):
    global processing_thread, stop_event, current_frame, current_live_stats
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
        "is_critical": False
    })

    processing_thread = threading.Thread(target=_process_video_stream_worker, args=(video_path, zone_polygons))
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

def _process_video_stream_worker(video_path, zone_polygons):
    global current_live_stats, current_frame, stop_event
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    model = YOLO(MODEL_NAME)
    tracker = sv.ByteTrack()

    # --- Setup Zones ---
    zones = [sv.PolygonZone(polygon=poly) for poly in zone_polygons]
    
    custom_colors = [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE]
    colors = sv.ColorPalette(colors=custom_colors)
    
    zone_annotators = [sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(i), thickness=2) for i, zone in enumerate(zones)]
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_position=sv.Position.TOP_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=100)
    
    previous_positions = defaultdict(lambda: None)
    
    calibration_data = {
        "avg_speeds": [], "total_counts": [],
        "zone_counts": [[] for _ in zone_polygons],
        "chaos_metrics": [] 
    }
    baseline_stats = {}
    frame_count = 0

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
        
        current_zone_counts = []
        for i, zone in enumerate(zones):
            mask = zone.trigger(detections=tracked_detections)
            current_zone_counts.append(np.sum(mask))

        annotated_frame = frame.copy() 

        # --- Calibration & Prediction Logic ---
        if frame_count < CALIBRATION_FRAMES:
            status_text = f"CALIBRATING... {frame_count}/{CALIBRATION_FRAMES}"
            status_color = (0, 255, 255) # Yellow

            # Collect data
            calibration_data["avg_speeds"].append(avg_speed)
            calibration_data["total_counts"].append(total_count)
            calibration_data["chaos_metrics"].append(chaos_metric) 
            for i in range(len(zones)):
                calibration_data["zone_counts"][i].append(current_zone_counts[i])

            if frame_count == CALIBRATION_FRAMES - 1:
                # Calculate all baselines
                baseline_stats["speed_mean"] = np.mean(calibration_data["avg_speeds"])
                baseline_stats["speed_std"] = np.std(calibration_data["avg_speeds"])
                baseline_stats["total_mean"] = np.mean(calibration_data["total_counts"])
                baseline_stats["total_std"] = np.std(calibration_data["total_counts"])
                baseline_stats["chaos_mean"] = np.mean(calibration_data["chaos_metrics"]) 
                baseline_stats["chaos_std"] = np.std(calibration_data["chaos_metrics"])   
                baseline_stats["zone_means"] = [np.mean(zone_data) if zone_data else 0 for zone_data in calibration_data["zone_counts"]]
                baseline_stats["zone_stds"] = [np.std(zone_data) if zone_data else 0 for zone_data in calibration_data["zone_counts"]]
                print("--- Calibration Complete ---")
        else:
            # --- Prediction Phase ---
            status_text = "NORMAL"
            status_color = (0, 255, 0) # Green
            is_warning = False
            is_critical = False

            # Check for anomalies
            speed_threshold_warn = baseline_stats["speed_mean"] + 2.0 * baseline_stats["speed_std"]
            speed_threshold_crit = baseline_stats["speed_mean"] + 3.5 * baseline_stats["speed_std"]
            total_threshold_warn = baseline_stats["total_mean"] + 2.5 * baseline_stats["total_std"]
            chaos_threshold_warn = baseline_stats["chaos_mean"] + 2.0 * baseline_stats["chaos_std"]
            
            if total_count > total_threshold_warn or avg_speed > speed_threshold_warn:
                is_warning = True
            if avg_speed > speed_threshold_crit and chaos_metric > chaos_threshold_warn:
                is_critical = True
                is_warning = True 

            for i in range(len(zones)):
                zone_count = current_zone_counts[i]
                zone_threshold_warn = baseline_stats["zone_means"][i] + 2.5 * baseline_stats["zone_stds"][i]
                if zone_count > zone_threshold_warn and zone_count > 5: 
                    is_warning = True

            if is_warning:
                status_text = "WARNING"
                status_color = (0, 255, 255) 
            if is_critical:
                status_text = "CRITICAL"
                status_color = (0, 0, 255) 
        
        global current_live_stats
        current_live_stats["total_count"] = total_count
        current_live_stats["avg_speed"] = avg_speed
        current_live_stats["chaos_metric"] = chaos_metric
        current_live_stats["status_text"] = status_text
        current_live_stats["is_warning"] = is_warning if 'is_warning' in locals() else False
        current_live_stats["is_critical"] = is_critical if 'is_critical' in locals() else False

        # --- Annotation ---
        # Draw Zones
        for i, zone in enumerate(zones):
            annotated_frame = zone_annotators[i].annotate(scene=annotated_frame)
            count_text = f"Zone {i+1}: {current_zone_counts[i]}"
            cv2.putText(annotated_frame, count_text, (50, 50 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(annotated_frame, count_text, (50, 50 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

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
    print("--- Video Stream Processing Finished ---")
from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
import os
from datetime import datetime

app = Flask(__name__)

import pygame
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

def start_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

helmet_model = YOLO(r"G:\Final project\HELMET\HELMET\best1.pt")
fire_model   = YOLO(r"G:\Final project\HELMET\HELMET\fire_model.pt")
vest_model   = None
mask_model   = None

DELAY_FIRE = 1.0
DELAY_HELMET = 2.5
DELAY_VEST = 2.5
DELAY_MASK = 2.5

ALERTS_DIR = (r"G:inal project\HELMET\HELMET\dashboard\static\alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)
alerts_list = []

def save_alert(frame, alert_type, severity):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    path = os.path.join(ALERTS_DIR, filename)
    cv2.imwrite(path, frame)
    alerts_list.append({
        "type": alert_type,
        "severity": severity,
        "time": datetime.now().strftime("%H:%M:%S"),
        "img": f"/static/alerts/{filename}"
    })

alerts_state = {
    "FIRE": {"active": False, "start": None},
    "NO_HELMET": {"active": False, "start": None},
    "NO_VEST": {"active": False, "start": None},
    "NO_MASK": {"active": False, "start": None},
}

cap = cv2.VideoCapture(0)

def generate_frames():
    global alerts_state
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frame = frame.copy()

        fire_results = fire_model(frame, conf=0.5, iou=0.3)
        frame = fire_results[0].plot()
        fire_flag = len(fire_results[0].boxes) > 0
        state = alerts_state["FIRE"]
        if fire_flag:
            if state["start"] is None:
                state["start"] = time.time()
            elif not state["active"] and time.time() - state["start"] >= DELAY_FIRE:
                threading.Thread(target=start_alarm, daemon=True).start()
                state["active"] = True
                save_alert(orig_frame, "FIRE", "CRITICAL")
        else:
            state["start"] = None
            if state["active"]:
                threading.Thread(target=stop_alarm, daemon=True).start()
                state["active"] = False


        helmet_results = helmet_model(frame, conf=0.5, iou=0.3)
        frame = helmet_results[0].plot()
        helmet_flag = len(helmet_results[0].boxes) > 0
        state = alerts_state["NO_HELMET"]
       
        if not helmet_flag:
            if state["start"] is None:
                state["start"] = time.time()
            elif not state["active"] and time.time() - state["start"] >= DELAY_HELMET:
                state["active"] = True
                save_alert(orig_frame, "NO_HELMET", "WARNING")
        else:
            state["start"] = None
            state["active"] = False

        state = alerts_state["NO_VEST"]
        if state["start"] is None:
            state["start"] = time.time()
        elif not state["active"] and time.time() - state["start"] >= DELAY_VEST:
            state["active"] = True
            save_alert(orig_frame, "NO_VEST", "WARNING")
                       
        else:
            state["start"] = None
            state["active"] = False

        state = alerts_state["NO_MASK"]
        if state["start"] is None:
            state["start"] = time.time()
        elif not state["active"] and time.time() - state["start"] >= DELAY_MASK:
            state["active"] = True
            save_alert(orig_frame, "NO_MASK", "WARNING")
        else:
            state["start"] = None
            state["active"] = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    return jsonify(alerts_list[-20:])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

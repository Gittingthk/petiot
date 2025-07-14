# -*- coding: utf-8 -*-
"""
Merged Version 2
================
Smart Pet Feeder PRO (Raspberry Pi)
• YOLOv8 + 5 × TFLite eye‑disease detectors
• Firebase (Realtime DB + Storage) integration
• Ultrasonic auto‑diagnosis trigger w/ cooldown
• Arduino stepper‑motor feeder support

All model/weight files are assumed to be inside **/home/taehun/pet-feeder-pro**.
"""

import os, time, cv2, logging, sys, select
import datetime as dt
from datetime import timezone
import numpy as np
import serial
import firebase_admin
from firebase_admin import credentials, db, storage
from picamera2 import Picamera2
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO

# ──────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR: str = "/home/taehun/pet-feeder-pro"          # <── 모델·가중치 위치
SAVE_DIR:  str = os.path.join(MODEL_DIR, "Picture")      # 결과 이미지 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

SERVICE_ACCOUNT_KEY_PATH = os.path.join(MODEL_DIR, "serviceAccountKey.json")
DATABASE_URL  = "https://my-pet-feeder-18354-default-rtdb.firebaseio.com/"
STORAGE_BUCKET = "my-pet-feeder-18354.firebasestorage.app"

ARDUINO_PORT = "/dev/ttyACM0"
BAUD_RATE    = 9600

ULTRASONIC_TRIG_PIN = 23      # BCM numbering
ULTRASONIC_ECHO_PIN = 24
DETECTION_DISTANCE_CM = 30    # cm threshold to trigger diagnosis
DIAGNOSIS_COOLDOWN_S  = 600   # seconds between auto‑diagnoses

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# 2. FIREBASE INIT
# ──────────────────────────────────────────────────────────────────────────────
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": DATABASE_URL,
    "storageBucket": STORAGE_BUCKET
})
commands_ref      = db.reference("commands")
device_status_ref = db.reference("device_status")
feeding_log_ref   = db.reference("feeding_log")
diagnosis_log_ref = db.reference("diagnosis_log")
bucket            = storage.bucket()

# ──────────────────────────────────────────────────────────────────────────────
# 3. HARDWARE INIT
# ──────────────────────────────────────────────────────────────────────────────
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    logging.info("Arduino connected on %s", ARDUINO_PORT)
except serial.SerialException as e:
    ser = None
    logging.warning("Arduino not connected: %s", e)

picam2 = Picamera2()
picam2.preview_configuration.main.size   = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start(); time.sleep(2)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ULTRASONIC_TRIG_PIN, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
GPIO.output(ULTRASONIC_TRIG_PIN, False)
time.sleep(1)

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL LOAD
# ──────────────────────────────────────────────────────────────────────────────
logging.info("Loading YOLOv8 model…")
yolo_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))

DISEASE_MODELS = {
    "cataract":        {"file": "cataract_multiclass_mobilenetv2_finetuned2.tflite",        "size": 96, "classes": ["Normal","Early","Immature","Mature"]},
    "conjunctivitis":  {"file": "conjunctivitis_mobilenetv2.tflite",                     "size": 96, "classes": ["No","Yes"]},
    "pigmentary_keratitis": {"file": "pigmentary_keratitis_mobilenetv2.tflite",          "size": 96, "classes": ["No","Yes"]},
    "nonulcerative_keratitis": {"file": "nonulcerative_keratitis_multiclass_mobilenetv2.tflite","size": 96,"classes": ["None","Mild","Severe"]},
    "ulcerative_keratitis":    {"file": "ulcerative_keratitis_multiclass_mobilenetv2.tflite",  "size": 96,"classes": ["None","Mild","Severe"]},
}

logging.info("Loading TFLite models…")
for name, cfg in DISEASE_MODELS.items():
    path = os.path.join(MODEL_DIR, cfg["file"])
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    cfg["interpreter"]   = interp
    cfg["input_details"] = interp.get_input_details()
    cfg["output_details"] = interp.get_output_details()
logging.info("All TFLite models loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. UTILS
# ──────────────────────────────────────────────────────────────────────────────

def utcnow() -> str:
    return dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB-xA) * max(0, yB-yA)
    areaA  = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB  = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def laplacian_sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def predict_tflite(cfg, crop):
    inp = cv2.resize(crop, (cfg["size"], cfg["size"])).astype(np.float32)/255.0
    inp = np.expand_dims(inp, 0)
    interp = cfg["interpreter"]
    interp.set_tensor(cfg["input_details"][0]["index"], inp)
    interp.invoke()
    out = interp.get_tensor(cfg["output_details"][0]["index"])[0]
    idx = int(out.argmax())
    return cfg["classes"][idx], float(out[idx])

def measure_distance():
    GPIO.output(ULTRASONIC_TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIG_PIN, False)
    start = time.time()
    pulse_start, pulse_end = start, start
    while GPIO.input(ULTRASONIC_ECHO_PIN)==0 and time.time()-start<0.1:
        pulse_start = time.time()
    while GPIO.input(ULTRASONIC_ECHO_PIN)==1 and time.time()-start<0.2:
        pulse_end = time.time()
    dur = pulse_end - pulse_start
    return (dur*34300)/2 if dur<0.02 else None

# ──────────────────────────────────────────────────────────────────────────────
# 6. CORE LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def run_eye_diagnosis():
    # Capture 3 frames, choose sharpest
    frames = [picam2.capture_array() for _ in range(3)]; time.sleep(0.5)
    img = frames[np.argmax([laplacian_sharpness(f) for f in frames])]

    yolo_res = yolo_model(img)[0]
    eyes = []
    for b in yolo_res.boxes:
        if yolo_model.names[int(b.cls[0])].find("eye")!=-1:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            conf = float(b.conf[0])
            eyes.append({"bbox":(x1,y1,x2,y2), "score":(x2-x1)*(y2-y1)*conf})
    eyes.sort(key=lambda e:e["score"], reverse=True)
    kept=[]
    for e in eyes:
        if all(compute_iou(e["bbox"],k["bbox"])<0.5 for k in kept):
            kept.append(e)
        if len(kept)>=2: break

    vis = img.copy(); summary=[]
    for e in kept:
        x1,y1,x2,y2 = e["bbox"]; crop = vis[y1:y2,x1:x2]
        y_offset=y1-10
        for name,cfg in DISEASE_MODELS.items():
            label,score = predict_tflite(cfg,crop)
            if label.lower() not in {"normal","none","no"}:
                txt=f"{name}:{label}({score*100:.1f}%)"; summary.append(txt)
                cv2.putText(vis,txt,(x1,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),2); y_offset-=20
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)

    fname = f"result_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    local = os.path.join(SAVE_DIR,fname)
    cv2.imwrite(local,vis)
    blob = bucket.blob(f"photos/{fname}"); blob.upload_from_filename(local); blob.make_public()
    url = blob.public_url
    diagnosis_log_ref.child("last_result").set({"result":summary,"image_url":url,"timestamp":utcnow()})
    logging.info("Diagnosis uploaded → %s", url)
    return summary, url

# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device_status_ref.update({"is_online":True,"current_state":"idle","last_seen":utcnow()})
    commands_ref.update({"feed":"NONE","diagnose":"NONE"})
    last_auto=0
    try:
        while True:
            device_status_ref.update({"last_seen":utcnow()})
            # A) ultrasonic auto trigger
            d=measure_distance()
            if d and d<DETECTION_DISTANCE_CM and time.time()-last_auto>DIAGNOSIS_COOLDOWN_S:
                logging.info("Auto‑diagnosis triggered (%.1f cm)", d)
                device_status_ref.update({"current_state":"diagnosing"})
                run_eye_diagnosis()
                last_auto=time.time()
                device_status_ref.update({"current_state":"idle"})
                continue
            # B) Firebase command
            cmds=commands_ref.get()
            if cmds.get("diagnose")!="NONE":
                logging.info("Firebase diagnose command received")
                device_status_ref.update({"current_state":"diagnosing"})
                run_eye_diagnosis()
                commands_ref.update({"diagnose":"NONE"})
                device_status_ref.update({"current_state":"idle"})
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        if ser and ser.is_open: ser.close()
        GPIO.cleanup()
        device_status_ref.update({"is_online":False,"current_state":"offline"})
        logging.info("Shutdown complete")

if __name__=="__main__":
    main()

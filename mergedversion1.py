# -*- coding: utf-8 -*-
import os, time, cv2, sys, select, logging
import numpy as np
import serial
import firebase_admin
from firebase_admin import credentials, db, storage
from picamera2 import Picamera2
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from datetime import datetime, timezone

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Paths ===
MODEL_DIR = '/home/juheon/Desktop/project/'
SAVE_DIR = '/home/juheon/Desktop/project/Picture/'
os.makedirs(SAVE_DIR, exist_ok=True)

# === Firebase Setup ===
SERVICE_ACCOUNT_KEY_PATH = "/home/juheon/serviceAccountKey.json"
DATABASE_URL = 'https://your-project.firebaseio.com/'
STORAGE_BUCKET = 'your-project.appspot.com'

cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL,
    'storageBucket': STORAGE_BUCKET
})
device_status_ref = db.reference('device_status')
diagnosis_log_ref = db.reference('diagnosis_log')
commands_ref = db.reference('commands')
bucket = storage.bucket()

# === Disease Model Settings ===
DISEASE_MODELS = {
    'cataract': {'path': 'cataract_multiclass_mobilenetv2_finetuned2.tflite', 'input_size': 96,
                 'classes': ['Normal', 'Early', 'Immature', 'Mature']},
    'conjunctivitis': {'path': 'conjunctivitis_mobilenetv2.tflite', 'input_size': 96,
                       'classes': ['No', 'Yes']},
    'pigmentary_keratitis': {'path': 'pigmentary_keratitis_mobilenetv2.tflite', 'input_size': 96,
                             'classes': ['No', 'Yes']},
    'nonulcerative_keratitis': {'path': 'nonulcerative_keratitis_multiclass_mobilenetv2.tflite', 'input_size': 96,
                                'classes': ['None', 'Mild', 'Severe']},
    'ulcerative_keratitis': {'path': 'ulcerative_keratitis_multiclass_mobilenetv2.tflite', 'input_size': 96,
                             'classes': ['None', 'Mild', 'Severe']}
}

# === Load YOLO & TFLite ===
logging.info("Loading YOLO...")
yolo_model = YOLO(os.path.join(MODEL_DIR, 'best.pt'))
logging.info("Loading TFLite models...")
for k, v in DISEASE_MODELS.items():
    model_path = os.path.join(MODEL_DIR, v['path'])
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    v['interpreter'] = interpreter
    v['input_details'] = interpreter.get_input_details()
    v['output_details'] = interpreter.get_output_details()
logging.info("All models loaded.")

# === PiCamera2 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)

# === Arduino Serial ===
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    logging.info("Serial connected on /dev/ttyACM0")
except:
    ser = None
    logging.warning("Serial not connected")

# === Ultrasonic Sensor ===
ULTRASONIC_TRIG_PIN = 23
ULTRASONIC_ECHO_PIN = 24
DETECTION_DISTANCE_CM = 30
DIAGNOSIS_COOLDOWN_S = 600
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ULTRASONIC_TRIG_PIN, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
GPIO.output(ULTRASONIC_TRIG_PIN, False)
time.sleep(1)
last_auto_diagnosis_time = 0

# === Utils ===
def get_utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def predict_tflite(model_info, image):
    resized = cv2.resize(image, (model_info['input_size'], model_info['input_size']))
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    interpreter = model_info['interpreter']
    interpreter.set_tensor(model_info['input_details'][0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(model_info['output_details'][0]['index'])[0]
    idx = int(np.argmax(output))
    score = float(output[idx])
    return model_info['classes'][idx], score

def measure_distance():
    GPIO.output(ULTRASONIC_TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIG_PIN, False)
    start_time = time.time()
    pulse_start, pulse_end = start_time, start_time
    while GPIO.input(ULTRASONIC_ECHO_PIN) == 0 and time.time() - start_time < 0.1:
        pulse_start = time.time()
    while GPIO.input(ULTRASONIC_ECHO_PIN) == 1 and time.time() - start_time < 0.2:
        pulse_end = time.time()
    duration = pulse_end - pulse_start
    return (duration * 34300) / 2 if duration < 0.02 else None

def capture_and_diagnose():
    frames = [picam2.capture_array() for _ in range(3)]
    time.sleep(0.5)
    sharpest = frames[np.argmax([sharpness(f) for f in frames])]
    results = yolo_model(sharpest)[0]
    detections = []
    for box in results.boxes:
        label = yolo_model.names[int(box.cls[0])]
        if 'eye' not in label: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append({'bbox': (x1, y1, x2, y2), 'score': (x2 - x1)*(y2 - y1) * conf})
    detections.sort(key=lambda x: x['score'], reverse=True)
    filtered = []
    for det in detections:
        if all(compute_iou(det['bbox'], keep['bbox']) < 0.5 for keep in filtered):
            filtered.append(det)
        if len(filtered) >= 2: break

    vis = sharpest.copy()
    results_text = []
    for i, det in enumerate(filtered):
        x1, y1, x2, y2 = det['bbox']
        crop = vis[y1:y2, x1:x2]
        y_offset = y1 - 10
        for name, model in DISEASE_MODELS.items():
            label, score = predict_tflite(model, crop)
            if label.lower() not in ['normal', 'none', 'no']:
                text = f"{name}: {label} ({score*100:.1f}%)"
                cv2.putText(vis, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255),2)
                y_offset -= 20
                results_text.append(text)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

    # Upload
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    local_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(local_path, vis)
    blob = bucket.blob(f'photos/{filename}')
    blob.upload_from_filename(local_path)
    blob.make_public()
    image_url = blob.public_url
    diagnosis_log_ref.child('last_result').set({
        'result': results_text,
        'image_url': image_url,
        'timestamp': get_utc_timestamp()
    })
    logging.info(f"Saved & uploaded diagnosis image: {image_url}")
    return results_text, image_url

# === MAIN LOOP ===
device_status_ref.update({'is_online': True, 'current_state': 'idle', 'last_seen': get_utc_timestamp()})
commands_ref.update({'diagnose': 'NONE'})

try:
    while True:
        device_status_ref.update({'last_seen': get_utc_timestamp()})
        distance = measure_distance()
        if distance and distance < DETECTION_DISTANCE_CM and (time.time() - last_auto_diagnosis_time) > DIAGNOSIS_COOLDOWN_S:
            logging.info(f"Auto trigger: detected at {distance:.1f}cm")
            device_status_ref.update({'current_state': 'diagnosing'})
            capture_and_diagnose()
            last_auto_diagnosis_time = time.time()
            device_status_ref.update({'current_state': 'idle'})
            continue

        commands = commands_ref.get()
        if commands.get('diagnose') != 'NONE':
            logging.info("Firebase trigger: diagnose")
            device_status_ref.update({'current_state': 'diagnosing'})
            capture_and_diagnose()
            commands_ref.update({'diagnose': 'NONE'})
            device_status_ref.update({'current_state': 'idle'})
        time.sleep(0.5)
except KeyboardInterrupt:
    logging.info("Interrupted by user.")
finally:
    if ser: ser.close()
    GPIO.cleanup()
    device_status_ref.update({'is_online': False, 'current_state': 'offline'})
    logging.info("Shutdown complete.")

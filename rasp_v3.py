# -*- coding: utf-8 -*-
"""
스마트 펫 피더 PRO (Raspberry Pi v3)
──────────────────────────────────────────────────────────────────────────────
• YOLOv8 + 5 × TFLite 안구 질환 진단
• Firebase Realtime DB / Storage 통합
• 초음파 기반 자동 진단 (쿨다운 포함)
• Arduino 스텝모터 사료 급여 (수동)

데이터 흐름
──────────
1) 수동 급여
   ─ Flutter 앱 → commands/feed 노드(dict) 작성
   ─ Pi → Arduino 로 "FEED:<g>" 전송 후 완료 신호 대기
   ─ Pi → feeding_log, device_status 업데이트
2) 자동 진단
   ─ 초음파 감지 또는 commands/diagnose 트리거
   ─ Pi → 카메라 촬영 & 모델 추론 → Storage 업로드
   ─ Pi → diagnosis_log, device_status 업데이트
"""

import os, time, cv2, sys, select, logging, json, datetime as dt
from datetime import timezone
import numpy as np
import serial
import firebase_admin
from firebase_admin import credentials, db, storage
from picamera2 import Picamera2
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. 경로 · 상수
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/taehun/pet-feeder-pro"
SAVE_DIR  = os.path.join(MODEL_DIR, "Picture")
os.makedirs(SAVE_DIR, exist_ok=True)

SERVICE_KEY = os.path.join(MODEL_DIR, "serviceAccountKey.json")
DB_URL      = "https://my-pet-feeder-18354-default-rtdb.firebaseio.com/"
BUCKET_NAME = "my-pet-feeder-18354.appspot.com"

ARDUINO_PORT = "/dev/ttyACM0"; BAUD = 9600
ULTRA_TRIG, ULTRA_ECHO = 23, 24  # BCM 핀 번호
DIST_THRESHOLD_CM = 30
DIAG_COOLDOWN_S   = 600
SER_FEED_TIMEOUT_S = 30

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Firebase 초기화
# ─────────────────────────────────────────────────────────────────────────────
cred = credentials.Certificate(SERVICE_KEY)
firebase_admin.initialize_app(cred, {
    "databaseURL": DB_URL,
    "storageBucket": BUCKET_NAME
})
commands_ref      = db.reference("commands")
status_ref        = db.reference("device_status")
feed_log_ref      = db.reference("feeding_log")
diagnosis_log_ref = db.reference("diagnosis_log")
bucket           = storage.bucket()

# ─────────────────────────────────────────────────────────────────────────────
# 3. 하드웨어 초기화
# ─────────────────────────────────────────────────────────────────────────────
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)
    time.sleep(2)
    logging.info("Arduino 연결 완료 (%s)", ARDUINO_PORT)
except serial.SerialException as e:
    ser = None
    logging.warning("Arduino 연결 실패: %s", e)

picam2 = Picamera2()
picam2.preview_configuration.main.size   = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview"); picam2.start(); time.sleep(2)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ULTRA_TRIG, GPIO.OUT)
GPIO.setup(ULTRA_ECHO, GPIO.IN)
GPIO.output(ULTRA_TRIG, False); time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 모델 로드
# ─────────────────────────────────────────────────────────────────────────────
logging.info("YOLOv8 로딩 중…")
yolo = YOLO(os.path.join(MODEL_DIR, "best.pt"))

DISEASE_CFG: Dict[str, Dict[str, Any]] = {
    "cataract": {"file": "cataract_multiclass_mobilenetv2_finetuned2.tflite", "size": 96, "classes": ["Normal","Early","Immature","Mature"]},
    "conjunctivitis": {"file": "conjunctivitis_mobilenetv2.tflite", "size": 96, "classes": ["No","Yes"]},
    "pigmentary_keratitis": {"file": "pigmentary_keratitis_mobilenetv2.tflite", "size": 96, "classes": ["No","Yes"]},
    "nonulcerative_keratitis": {"file": "nonulcerative_keratitis_multiclass_mobilenetv2.tflite", "size": 96, "classes": ["None","Mild","Severe"]},
    "ulcerative_keratitis": {"file": "ulcerative_keratitis_multiclass_mobilenetv2.tflite", "size": 96, "classes": ["None","Mild","Severe"]},
}

logging.info("TFLite 모델 로딩 중…")
for cfg in DISEASE_CFG.values():
    path = os.path.join(MODEL_DIR, cfg["file"])
    ip = tflite.Interpreter(model_path=path); ip.allocate_tensors()
    cfg.update({"interpreter": ip,
                "inp": ip.get_input_details(),
                "out": ip.get_output_details()})
logging.info("모든 모델 로드 완료")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────

def utcnow() -> str:
    """UTC ISO‑8601 문자열 반환"""
    return dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def laplacian_var(img):
    """이미지 샤프니스 측정 (라플라시안 분산)"""
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def predict(cfg, crop):
    """TFLite 추론 & (라벨, 확률) 반환"""
    x = cv2.resize(crop, (cfg["size"], cfg["size"])).astype(np.float32)/255.0
    x = np.expand_dims(x, 0)
    ip = cfg["interpreter"]
    ip.set_tensor(cfg["inp"][0]["index"], x)
    ip.invoke()
    y = ip.get_tensor(cfg["out"][0]["index"])[0]
    idx = int(y.argmax()); return cfg["classes"][idx], float(y[idx])


def distance_cm():
    GPIO.output(ULTRA_TRIG, True); time.sleep(1e-5); GPIO.output(ULTRA_TRIG, False)
    start = time.time(); pulse_start, pulse_end = start, start
    while GPIO.input(ULTRA_ECHO) == 0 and time.time()-start < .1: pulse_start = time.time()
    while GPIO.input(ULTRA_ECHO) == 1 and time.time()-start < .2: pulse_end  = time.time()
    dur = pulse_end - pulse_start; return (dur*34300)/2 if dur < .02 else None


def update_status(**kwargs):
    status_ref.update({**kwargs, "last_seen": utcnow()})

# ─────────────────────────────────────────────────────────────────────────────
# 6. 핵심 기능
# ─────────────────────────────────────────────────────────────────────────────

def diagnose():
    """카메라 촬영 → 눈 검출 → 질환 추론 → Storage 업로드"""
    frames = [picam2.capture_array() for _ in range(3)]; time.sleep(.5)
    img = frames[np.argmax([laplacian_var(f) for f in frames])]

    eyes = []
    for b in yolo(img)[0].boxes:
        if yolo.names[int(b.cls[0])].find("eye") != -1:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            eyes.append({"bbox": (x1,y1,x2,y2), "score": (x2-x1)*(y2-y1)*conf})
    eyes.sort(key=lambda e: e["score"], reverse=True)

    kept = []
    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA)*max(0, yB-yA)
        areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA+areaB-inter+1e-6)

    for e in eyes:
        if all(iou(e["bbox"], k["bbox"]) < .5 for k in kept): kept.append(e)
        if len(kept) >= 2: break

    vis = img.copy(); summary = []
    for e in kept:
        x1,y1,x2,y2 = e["bbox"]
        crop = vis[y1:y2, x1:x2]
        y_offset = y1 - 10
        for name, cfg in DISEASE_CFG.items():
            lbl, prob = predict(cfg, crop)
            if lbl.lower() not in {"normal","none","no"}:
                txt = f"{name}:{lbl}({prob*100:.1f}%)"; summary.append(txt)
                cv2.putText(vis, txt, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 2); y_offset -= 20
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

    fname = f"result_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    local = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(local, vis)
    blob = bucket.blob(f"photos/{fname}"); blob.upload_from_filename(local); blob.make_public()

    diagnosis_log_ref.child("last_result").set({
        "result": summary,
        "image_url": blob.public_url,
        "timestamp": utcnow()
    })
    logging.info("진단 결과 업로드 → %s", blob.public_url)


def feed(amount_g: int):
    """스텝모터로 사료 급여 (g 단위)"""
    if amount_g <= 0:
        logging.warning("잘못된 급여량: %d g", amount_g); return False
    if not ser:
        logging.warning("Arduino 미연결 – 시뮬레이션 진행"); time.sleep(2); return True

    cmd = f"FEED:{amount_g}\n".encode()
    ser.reset_input_buffer(); ser.write(cmd)
    logging.info("Arduino ← %s", cmd.decode().strip())

    start = time.time()
    while time.time() - start < SER_FEED_TIMEOUT_S:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", "ignore").strip()
            if line == "FED_DONE":
                logging.info("급여 완료 신호 수신"); return True
    logging.error("급여 완료 신호 미수신"); return False

# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 루프
# ─────────────────────────────────────────────────────────────────────────────

def main():
    update_status(is_online=True, current_state="idle")
    commands_ref.set({"feed": "NONE", "diagnose": "NONE"})
    last_diag_auto = 0

    try:
        while True:
            update_status()  # heartbeat

            # (A) 초음파 → 자동 진단
            d = distance_cm()
            if d and d < DIST_THRESHOLD_CM and time.time() - last_diag_auto > DIAG_COOLDOWN_S:
                logging.info("자동 진단 트리거 (%.1f cm)", d)
                update_status(current_state="diagnosing")
                diagnose(); last_diag_auto = time.time()
                update_status(current_state="idle")
                continue

            # (B) Firebase 명령 체크
            cmds = commands_ref.get()
            # 1) 피드 명령
            if isinstance(cmds.get("feed"), dict):
                req = cmds["feed"]; amount = int(req.get("amount_g", 0))
                logging.info("피드 명령 수신: %dg", amount)
                update_status(current_state="feeding")
                ok = feed(amount)
                if ok:
                    feed_log_ref.push({"amount_g": amount, "timestamp": utcnow()})
                commands_ref.update({"feed": "NONE"})
                update_status(current_state="idle")
                continue
            # 2) 진단 명령
            if cmds.get("diagnose") == "REQUEST":
                logging.info("진단 명령 수신")
                update_status(current_state="diagnosing")
                diagnose(); commands_ref.update({"diagnose": "NONE"})
                update_status(current_state="idle")
            time.sleep(.5)
    except KeyboardInterrupt:
        logging.info("사용자 중단")
    finally:
        if ser and ser.is_open: ser.close()
        GPIO.cleanup()
        update_status(is_online=False, current_state="offline")
        logging.info("종료 완료")


if __name__ == "__main__":
    main()

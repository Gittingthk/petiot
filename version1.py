# ==============================================================================
# Smart Pet Feeder PRO - Main Controller (Raspberry Pi)
# [FINAL VERSION: Ultrasonic Auto-Diagnosis & Stepper Motor Control]
# ==============================================================================
# - Automatically diagnoses when a pet is detected by the ultrasonic sensor.
# - Listens for manual Firebase commands ('feed' and 'diagnose').
# - Commands an Arduino via USB Serial to dispense food using a stepper motor.
# - Captures image using Picamera2, detects dog with YOLOv8.
# - Performs a simulated health check and uploads results to Firebase.
# ==============================================================================

import time
import datetime
import logging
import serial
import cv2
import firebase_admin
from firebase_admin import credentials, db, storage
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO ### NEW ### For ultrasonic sensor

# --- 1. CONFIGURATION & INITIALIZATION ---

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- User-defined Constants ---
# IMPORTANT: Update these paths and URLs with your own information.
SERVICE_ACCOUNT_KEY_PATH = "/home/taehun/pet-feeder-pro/serviceAccountKey.json"
DATABASE_URL = 'https://my-pet-feeder-18354-default-rtdb.firebaseio.com/'
STORAGE_BUCKET = 'my-pet-feeder-18354.firebasestorage.app'

# --- Hardware Pins & Settings ---
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
### NEW ### Ultrasonic Sensor GPIO Pins (BCM numbering)
ULTRASONIC_TRIG_PIN = 23
ULTRASONIC_ECHO_PIN = 24

### NEW ### Auto-Diagnosis Logic Settings
# 이 거리(cm) 안으로 들어오면 자동 진단 시작
DETECTION_DISTANCE_CM = 30
# 자동 진단 후 재진단 방지를 위한 최소 대기 시간 (초). (예: 600초 = 10분)
DIAGNOSIS_COOLDOWN_S = 600

# --- Firebase Initialization ---
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL, 'storageBucket': STORAGE_BUCKET})
    
    commands_ref = db.reference('commands')
    device_status_ref = db.reference('device_status')
    feeding_log_ref = db.reference('feeding_log')
    diagnosis_log_ref = db.reference('diagnosis_log')
    bucket = storage.bucket()
    
    logging.info("Firebase SDK initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Firebase: {e}")
    exit()

# --- Hardware Initialization ---
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Arduino가 재부팅될 시간을 줍니다.
    logging.info(f"Successfully connected to Arduino on {ARDUINO_PORT}.")
except serial.SerialException as e:
    ser = None
    logging.warning(f"Could not connect to Arduino: {e}. Running in test mode (no feeding).")

try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    logging.info("Picamera2 initialized successfully.")
except Exception as e:
    picam2 = None
    logging.warning(f"Could not initialize camera: {e}. Diagnosis will be simulated.")

try:
    yolo_model = YOLO('yolov8n.pt')
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    yolo_model = None
    logging.warning(f"Could not load YOLOv8 model: {e}. Diagnosis will be simulated.")

### NEW ### Ultrasonic Sensor GPIO Setup
try:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ULTRASONIC_TRIG_PIN, GPIO.OUT)
    GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
    GPIO.output(ULTRASONIC_TRIG_PIN, False)
    logging.info("Ultrasonic sensor GPIO initialized.")
    time.sleep(1) # 센서 안정화 시간
except Exception as e:
    logging.error(f"Failed to initialize GPIO for ultrasonic sensor: {e}")
    # GPIO 없이도 계속 실행되도록 설정
    GPIO.cleanup()


# --- 2. HELPER & CORE FUNCTIONS ---

def get_utc_timestamp():
    """Returns the current time in UTC ISO 8601 format compatible with the app."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')

### NEW ### Function to measure distance with HC-SR04
def measure_distance():
    """Measures distance using the ultrasonic sensor and returns it in cm."""
    try:
        # Send a 10us pulse to trigger
        GPIO.output(ULTRASONIC_TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(ULTRASONIC_TRIG_PIN, False)

        pulse_start_time = time.time()
        pulse_end_time = time.time()
        
        timeout_start = time.time()
        # Save pulse start time
        while GPIO.input(ULTRASONIC_ECHO_PIN) == 0:
            pulse_start_time = time.time()
            if pulse_start_time - timeout_start > 0.1: # 0.1초 이상 신호 없으면 타임아웃
                return None

        timeout_start = time.time()
        # Save pulse arrival time
        while GPIO.input(ULTRASONIC_ECHO_PIN) == 1:
            pulse_end_time = time.time()
            if pulse_end_time - timeout_start > 0.1: # 0.1초 이상 신호 지속되면 타임아웃
                return None

        pulse_duration = pulse_end_time - pulse_start_time
        # Speed of sound is ~34300 cm/s. Divide by 2 for round trip.
        distance = (pulse_duration * 34300) / 2
        return distance
    except Exception:
        # GPIO 관련 오류 발생 시 None 반환
        return None

### MODIFIED FOR STEPPER MOTOR ###
def feed_via_arduino(amount=1):
    """Commands the Arduino to dispense a specific amount of food."""
    logging.info(f"Sending 'FEED:{amount}' command to Arduino...")
    device_status_ref.update({'current_state': 'feeding'})

    if not ser:
        logging.warning("Arduino not connected. Simulating feed cycle.")
        time.sleep(3)
        return True

    try:
        # 스텝모터 제어를 위해 양(amount)을 포함한 명령 전송
        command_to_send = f"FEED:{amount}\n"
        ser.write(command_to_send.encode('utf-8')) # 문자열을 byte로 인코딩하여 전송
        
        start_time = time.time()
        # 스텝모터 동작 시간을 고려해 타임아웃을 30초로 늘림
        while time.time() - start_time < 30:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').strip()
                logging.info(f"Received from Arduino: '{response}'")
                if "DONE" in response:
                    logging.info("Feed cycle confirmed by Arduino.")
                    return True
        logging.warning("Timeout: No 'DONE' confirmation received from Arduino.")
        return False
    except Exception as e:
        logging.error(f"Error during serial communication with Arduino: {e}")
        return False

def diagnose_pet_health(dog_image):
    """Simulated health diagnosis."""
    import random
    logging.info("Simulating pet health diagnosis...")
    time.sleep(1)
    return "정상" if random.random() > 0.5 else "백내장 의심"

def capture_and_analyze():
    """Captures an image, runs analysis, uploads, and returns results."""
    logging.info("Starting capture and analysis process...")
    device_status_ref.update({'current_state': 'diagnosing'})

    if not picam2 or not yolo_model:
        logging.warning("Camera or YOLO model not available. Simulating analysis.")
        time.sleep(3)
        # 시뮬레이션 결과와 함께 유효한 이미지 URL 반환
        return "시뮬레이션: 정상", "https://firebasestorage.googleapis.com/v0/b/my-pet-feeder-18354.appspot.com/o/photos%2Fplaceholder.jpg?alt=media"

    # 1. Capture image
    picam2.start()
    time.sleep(2)
    image_array = picam2.capture_array()
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    picam2.stop()
    logging.info("Image captured.")

    # 2. Analyze image
    results = yolo_model(image_bgr)
    analysis_result = "반려동물 감지 실패"
    
    for r in results:
        for box in r.boxes:
            if r.names[int(box.cls[0])] == 'dog':
                logging.info("Dog detected in the image.")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dog_crop = image_bgr[y1:y2, x1:x2]
                analysis_result = diagnose_pet_health(dog_crop)
                
                box_color = (0, 255, 0) if "정상" in analysis_result else (0, 0, 255)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(image_bgr, analysis_result, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                break
        if "감지 실패" not in analysis_result:
            break

    # 3. Upload image to Firebase Storage
    result_image_path = f'/tmp/result_{int(time.time())}.jpg'
    cv2.imwrite(result_image_path, image_bgr)
    
    blob = bucket.blob(f'photos/{result_image_path.split("/")[-1]}')
    blob.upload_from_filename(result_image_path)
    blob.make_public()
    image_url = blob.public_url
    logging.info(f"Image uploaded to: {image_url}")

    return analysis_result, image_url

def cleanup():
    """Gracefully closes resources on script exit."""
    logging.info("Cleaning up resources...")
    if ser and ser.is_open:
        ser.close()
        logging.info("Serial port closed.")
    
    ### NEW ###
    GPIO.cleanup()
    logging.info("GPIO cleaned up.")
    
    device_status_ref.update({'is_online': False, 'current_state': 'offline'})
    logging.info("Device status set to offline. Shutdown complete.")


# --- 3. MAIN EXECUTION LOOP (MAJOR CHANGES HERE) ---

def main():
    """Monitors sensors and Firebase for commands, then acts accordingly."""
    logging.info("Main controller is running. Monitoring sensors and Firebase...")

    # On startup, initialize the device status and reset commands
    device_status_ref.update({
        'is_online': True,
        'current_state': 'idle',
        'last_seen': get_utc_timestamp()
    })
    commands_ref.update({'feed': 'NONE', 'diagnose': 'NONE'})
    
    # 마지막 자동 진단 시간을 기록하여 반복적인 진단을 방지
    last_auto_diagnosis_time = 0

    while True:
        try:
            # 1. Heartbeat: 주기적으로 온라인 상태와 마지막 확인 시간을 업데이트
            device_status_ref.update({'last_seen': get_utc_timestamp()})
            
            # 2. 현재 상태가 'idle'(대기)일 때만 새 작업을 시작
            current_state = device_status_ref.child('current_state').get()
            if current_state != 'idle':
                time.sleep(1) # 다른 작업 중이면 잠시 대기
                continue

            # --- A. 자동 진단 트리거 확인 (초음파 센서) ---
            distance = measure_distance()
            # 거리가 측정되고, 쿨다운 시간이 지났는지 확인
            if distance is not None and distance < DETECTION_DISTANCE_CM:
                logging.info(f"Object detected at {distance:.1f} cm.")
                is_cooldown_over = (time.time() - last_auto_diagnosis_time) > DIAGNOSIS_COOLDOWN_S
                if is_cooldown_over:
                    logging.info("Cooldown over. Starting automatic diagnosis.")
                    result_text, result_url = capture_and_analyze() # 진단 실행
                    
                    if "감지 실패" not in result_text:
                        diagnosis_log_ref.child('last_result').set({
                            'result': result_text,
                            'image_url': result_url,
                            'timestamp': get_utc_timestamp()
                        })
                        logging.info("Automatic diagnosis log updated.")
                    else:
                        logging.info("Automatic diagnosis triggered, but no pet detected in image. Skipping log update.")
                        
                    last_auto_diagnosis_time = time.time() # 쿨다운 타이머 리셋
                    device_status_ref.update({'current_state': 'idle'}) # 상태를 즉시 idle로 복귀
                    continue # 자동 진단 후 루프 처음으로 돌아가 재확인

            # --- B. 수동 명령 확인 (Firebase) ---
            commands = commands_ref.get()
            if not commands:
                time.sleep(0.5) # 명령이 없으면 짧게 대기
                continue

            # (i) 수동 급식 명령 처리
            if commands.get('feed') and commands['feed'] != 'NONE':
                logging.info(f"='feed' command received with value: {commands['feed']}. Starting process.=")
                
                try: # 앱에서 숫자 대신 문자열(예: 'start')을 보낼 경우를 대비
                    amount = int(commands['feed']) if str(commands['feed']).isdigit() else 1
                except (ValueError, TypeError):
                    amount = 1 # 변환 실패 시 기본값 1
                
                if feed_via_arduino(amount=amount):
                    feeding_log_ref.update({'last_fed_timestamp': get_utc_timestamp()})
                    logging.info("Feeding log updated.")
                else:
                    logging.error("Feeding process failed.")
                
                device_status_ref.update({'current_state': 'idle'})
                commands_ref.update({'feed': 'NONE'})

            # (ii) 수동 진단 명령 처리
            elif commands.get('diagnose') and commands['diagnose'] != 'NONE':
                logging.info("='diagnose' command received. Starting process.=")
                result_text, result_url = capture_and_analyze()
                
                diagnosis_log_ref.child('last_result').set({
                    'result': result_text,
                    'image_url': result_url,
                    'timestamp': get_utc_timestamp()
                })
                logging.info("Manual diagnosis log updated.")
                
                device_status_ref.update({'current_state': 'idle'})
                commands_ref.update({'diagnose': 'NONE'})

            # 센서 감지와 네트워크 부하 사이의 균형을 맞추기 위한 대기
            time.sleep(0.5)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Exiting.")
            break
        except Exception as e:
            # 에러 발생 시 로그를 남기고 잠시 대기 후 재시도
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            device_status_ref.update({'current_state': 'error'})
            time.sleep(10)
            device_status_ref.update({'current_state': 'idle'}) # 에러 상태를 리셋하여 다시 시도
    
    cleanup()

if __name__ == "__main__":
    main()

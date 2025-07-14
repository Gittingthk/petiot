// ──────────────────────────────────────────────────────────────
// Pet Feeder Arduino Sketch v1.3  (Uno 호환 컴파일 오류 Fix)
//   • "FEED:<g>" 직렬 명령 + 수동 버튼 트리거
//   • 상태 LED + (옵션) 서보 입구 개폐
//   • 대형/소형 스텝모터 프로파일 전환 (SMALL_STEPPER)
//   • IRAM_ATTR → AVR 호환 매크로 처리
//   • 서보 핀을 D6 로 변경(28BYJ‑48 IN2(D9) 충돌 방지)
// ──────────────────────────────────────────────────────────────

#include <AccelStepper.h>
#include <Servo.h>

// ─────────── 매크로·설정 스위치 ───────────
#define SMALL_STEPPER 0   // 1 → 28BYJ-48+ULN2003 / 0 → NEMA17+A4988

// AVR 컴파일러에는 IRAM_ATTR 정의가 없으므로 빈 매크로로 처리
#ifndef IRAM_ATTR
  #define IRAM_ATTR
#endif

// ─ 핀 정의 ─
#if SMALL_STEPPER
  // 28BYJ-48 + ULN2003 IN1~4 (시계방향)
  const int IN1 = 8, IN2 = 9, IN3 = 10, IN4 = 11;
  AccelStepper stepper(AccelStepper::FULL4WIRE, IN1, IN3, IN2, IN4);
#else
  const int PIN_STEP = 2;
  const int PIN_DIR  = 3;
  const int PIN_EN   = 12;
  AccelStepper stepper(AccelStepper::DRIVER, PIN_STEP, PIN_DIR);
#endif

const int PIN_LED   = 5;   // 상태 LED (HIGH=ON)
const int PIN_BTN   = 7;   // 수동 급여 버튼 (INPUT_PULLUP)
const int PIN_SERVO = 6;   // 서보 PWM (D6, PWM) ─ D9 충돌 방지

// ─ 모터·사료 파라미터 ─
#if SMALL_STEPPER
  const float STEPS_PER_REV = 2048.0;  // 28BYJ 기어비 포함
#else
  const float STEPS_PER_REV = 200.0;   // 1.8° NEMA17
#endif

const float LEAD_MM_PER_REV = 8.0;     // 리드(mm) – 측정 후 보정
const float GRAMS_PER_MM    = 1.5;     // 1 mm 이동 시 배출 g – 실측 후 보정
const float STEPS_PER_GRAM  = (STEPS_PER_REV / LEAD_MM_PER_REV) * GRAMS_PER_MM;

const int MANUAL_FEED_G = 20;          // 수동 버튼 급여량(g)

// ─ 서보 각도 설정 ─
const int SERVO_OPEN  = 90;
const int SERVO_CLOSE = 0;
Servo gateServo;

// ─ 상태 변수 ─
String serialBuffer;
volatile bool btnPressed = false;
bool isFeeding = false;

// 버튼 인터럽트 ISR
void IRAM_ATTR onButton() {
  btnPressed = true;
}

void setup() {
  Serial.begin(9600);

#if !SMALL_STEPPER
  pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_EN, LOW); // 드라이버 활성
#endif

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  pinMode(PIN_BTN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_BTN), onButton, FALLING);

  stepper.setMaxSpeed(SMALL_STEPPER ? 800 : 1200);
  stepper.setAcceleration(SMALL_STEPPER ? 800 : 2000);

  gateServo.attach(PIN_SERVO);
  gateServo.write(SERVO_CLOSE);

  Serial.println("Arduino Ready (v1.3)");
}

void gateOpen()  { gateServo.write(SERVO_OPEN);  delay(300); }
void gateClose() { gateServo.write(SERVO_CLOSE); delay(300); }

void dispense(int grams) {
  if (grams <= 0) return;
  isFeeding = true;
  digitalWrite(PIN_LED, HIGH);
  gateOpen();

  long steps = (long)(grams * STEPS_PER_GRAM);
  stepper.move(steps);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }

  delay(200);
  gateClose();
  digitalWrite(PIN_LED, LOW);
  isFeeding = false;
  Serial.println("FED_DONE");
}

void handleCommand(String cmd) {
  cmd.trim(); cmd.toUpperCase();
  if (cmd.startsWith("FEED:")) {
    int amount = cmd.substring(5).toInt();
    Serial.print("RECV FEED → "); Serial.println(amount);
    dispense(amount);
  } else {
    Serial.print("UNKNOWN CMD: "); Serial.println(cmd);
  }
}

void loop() {
  // 직렬 수신
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length()) { handleCommand(serialBuffer); serialBuffer = ""; }
    } else serialBuffer += c;
  }

  // 수동 버튼 급여
  if (btnPressed && !isFeeding) {
    btnPressed = false;
    dispense(MANUAL_FEED_G);
  }
}

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN  125 // ~500 us (0 deg)
#define SERVOMAX  575 // ~2400 us (180 deg)

// --- HARD SAFETY LIMITS ---
const int MIN_SAFE_ANGLE = 55;
const int MAX_SAFE_ANGLE = 85;
const int NEUTRAL_ANGLE  = 70;

// --- SPEED CONTROL ---
const float STEP_SIZE = 4.0;
const unsigned long UPDATE_INTERVAL_MS = 10;

float currentAngle[3] = {NEUTRAL_ANGLE, NEUTRAL_ANGLE, NEUTRAL_ANGLE};
float targetAngle[3]  = {NEUTRAL_ANGLE, NEUTRAL_ANGLE, NEUTRAL_ANGLE};
unsigned long lastUpdate = 0;

// 1. Updated to float math (preserves your inverted SERVOMAX -> SERVOMIN mapping)
int angleToPulse(float angle) {
  angle = constrain(angle, 0.0, 180.0);
  return (int)round(SERVOMAX - (angle / 180.0) * (SERVOMAX - SERVOMIN));
}

// 2. Changed angle parameter from int to float
void setArm(uint8_t channel, float angle) {
  pwm.setPWM(channel, 0, angleToPulse(angle));
}

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  for (int ch = 0; ch < 3; ch++) {
    setArm(ch, NEUTRAL_ANGLE);
  }
}

void loop() {
  // 3. Decode 0..250 byte back to 45.0..95.0 deg (0.2 deg steps)
  if (Serial.available() >= 4) {
    if (Serial.read() == 0xFF) {
      for (int ch = 0; ch < 3; ch++) {
        int rawByte = Serial.read();
        float decodedAngle = 45.0 + (rawByte / 5.0);
        targetAngle[ch] = constrain(decodedAngle, (float)MIN_SAFE_ANGLE, (float)MAX_SAFE_ANGLE);
      }
    }
  }

  // 4. Smooth interpolation without integer rounding
  unsigned long now = millis();
  if (now - lastUpdate >= UPDATE_INTERVAL_MS) {
    lastUpdate = now;

    for (int ch = 0; ch < 3; ch++) {
      float diff = targetAngle[ch] - currentAngle[ch];

      if (abs(diff) > 0.05) {
        if (diff > STEP_SIZE) {
          currentAngle[ch] += STEP_SIZE;
        } else if (diff < -STEP_SIZE) {
          currentAngle[ch] -= STEP_SIZE;
        } else {
          currentAngle[ch] = targetAngle[ch];
        }
        setArm(ch, currentAngle[ch]); // Removed (int)round() so 0.2 deg steps reach the PWM driver
      }
    }
  }
}
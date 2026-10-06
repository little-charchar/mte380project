#include <Servo.h>

// Servo configuration
Servo beamServo;
const int servoPin = 9;
const int MIN_ANGLE = 92;   // was 0; level 122 +/- 30 on our linkage
const int MAX_ANGLE = 152;  // was 30
const int NEUTRAL_ANGLE = 122;  // was 15; beam is level at 122 (final mechanism)
const int MAX_STEP = 250;  // finer resolution: byte 0..250 maps to MIN_ANGLE..MAX_ANGLE (0.24 deg steps)

// Convert angle (deg) to pulse width the same way Servo.write() does, but without rounding to whole degrees
int angleToUs(float angle) { return (int)(544.0 + angle * (2400.0 - 544.0) / 180.0 + 0.5); }

// Communication variables
int targetUs = angleToUs(NEUTRAL_ANGLE);
bool newCommand = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo
  beamServo.attach(servoPin);
  
  // Move to neutral position
  beamServo.write(NEUTRAL_ANGLE);
  
  // Initialize variables
  targetUs = angleToUs(NEUTRAL_ANGLE);
  
  // Optional: Send ready message
  Serial.println("Arduino servo controller ready");
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    // Read the incoming byte
    int receivedStep = Serial.read();
    
    // Validate step range, convert step -> angle -> pulse width
    if (receivedStep >= 0 && receivedStep <= MAX_STEP) {
      targetUs = angleToUs(MIN_ANGLE + receivedStep * (float)(MAX_ANGLE - MIN_ANGLE) / MAX_STEP);
      newCommand = true;
    }
  }
  
  // Update servo if new command received
  if (newCommand) {
    beamServo.writeMicroseconds(targetUs);
    newCommand = false;
    
    // Optional: Echo back the angle for debugging
    // Serial.print("Angle set to: ");
    // Serial.println(targetUs);
  }
  
  // Small delay for stability
  delay(10);
}
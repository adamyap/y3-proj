#include <Servo_Hardware_PWM.h>

Servo servoX;
Servo servoY;
int pos = 100; // servo position

void setup() {
  servoX.attach(2); // attach servos to designated pins
  servoY.attach(3);
  Serial.begin(9600);  // opens serial port, 9600 baud
  Serial.setTimeout(10);
  servoX.write(pos);  // write position to servos
  servoY.write(pos);
}

void loop() {
  if (Serial.available()) {
    char motor = Serial.read(); // read the motor identifier
    int angle = Serial.parseInt(); // read the angle
    pos = angle + 40;  // add 40 to value (so to not use the full range of the servo)
    // Ensure the angle is within the range for your servos
    pos = constrain(pos, 0, 180);
    // Move the corresponding servo
    if (motor == 'x') {
      servoX.write(pos);
    } else if (motor == 'y') {
      servoY.write(pos);
    }
  }
}
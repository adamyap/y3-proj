#include <Servo.h>

Servo servo1;  // create servo object to control a servo
Servo servo2;
int pos1 = 100;    // variable to store the servo position
int pos2 = 100;

void setup() {
  servo1.attach(11);  // attaches the servo on pin 6 to the servo object
  servo2.attach(3);
  Serial.begin(9600);  // opens serial port, sets data rate to 9600 bps
  Serial.setTimeout(10);
  servo1.write(pos1);  // move the servo to the initial position
  servo2.write(pos2);
}

void loop() {
  if (Serial.available()) {
    int val1 = Serial.parseInt();  // read the incoming number for servo1
    if (Serial.read() != ',') return;  // check for the comma
    int val2 = Serial.parseInt();  // read the incoming number for servo2
    if (val1 >= 0 && val1 <= 120) {  // check if the value is within the expected range
      pos1 = val1 + 40;  // add 40 to the value
      servo1.write(pos1);  // tell servo1 to go to the position
      //Serial.print(val1);
      //Serial.println(" degrees for servo1.");
    }
    if (val2 >= 0 && val2 <= 120) {  // check if the value is within the expected range
      pos2 = val2 + 40;  // add 40 to the value
      servo2.write(pos2);  // tell servo2 to go to the position
      //Serial.print(val2);
      //Serial.println(" degrees for servo2.");
    }
  }
}
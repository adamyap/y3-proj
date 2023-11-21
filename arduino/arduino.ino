#include <Servo.h>

Servo servo1;
Servo servo2;
int pos1 = 100; // servo position
int pos2 = 100;

void setup() {
  servo1.attach(11); // attach servos to designated pins
  servo2.attach(3);
  Serial.begin(9600);  // opens serial port, 9600 baud
  Serial.setTimeout(10);
  servo1.write(pos1);  // write position to servos
  servo2.write(pos2);
}

void loop() {
  if (Serial.available()) {
    int input1 = Serial.parseInt();  // read incoming serial number
    if (Serial.read() != ',') return;  // check for comma
    int input2 = Serial.parseInt(); 
    if (input1 >= 0 && input1 <= 120) {  // check if value within expected range
      pos1 = input1 + 40;  // add 40 to value (so to not use the full range of the servo)
      servo1.write(pos1);
      //Serial.print(val1);
      //Serial.println(" degrees for servo1.");
    }
    if (input2 >= 0 && input2 <= 120) {
      pos2 = input2 + 40;
      servo2.write(pos2);
      //Serial.print(val2);
      //Serial.println(" degrees for servo2.");
    }
  }
}
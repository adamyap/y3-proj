#include <util/atomic.h> // For the ATOMIC_BLOCK macro

//Motor 1
#define ENCA 3 // YELLOW
#define ENCB 9 // WHITE
#define PWM 5
#define IN2 6
#define IN1 7

//Motor 2
#define ENCC 2//yellow
#define ENCD 8//white
#define PWM2 10
#define IN4 11
#define IN3 12

//Motor 1
volatile int posi = 0; // specify posi as volatile: https://www.arduino.cc/reference/en/language/variables/variable-scope-qualifiers/volatile/
long prevT = 0;
float eprev = 0;
float eintegral = 0;

//Motor 2
volatile int posi2 = 0; // specify posi as volatile: https://www.arduino.cc/reference/en/language/variables/variable-scope-qualifiers/volatile/
long prevT2 = 0;
float eprev2 = 0;
float eintegral2 = 0;


void setup() {
  Serial.begin(9600);
  //Motor 1
  pinMode(ENCA,INPUT);
  pinMode(ENCB,INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA),readEncoder,RISING);
  pinMode(PWM,OUTPUT);
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);

  //Motor 2
  pinMode(ENCC,INPUT);
  pinMode(ENCD,INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCC),readEncoder2,RISING);
  pinMode(PWM2,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);
  
  Serial.println("target pos");
}

void loop() {

  //MOTOR 1

  int target = 2100; // set target position
 
  // PID constants
  float kp = 1;
  float kd = 0.0;
  float ki = 0.0;

  long currT = micros();
  float deltaT = ((float) (currT - prevT))/( 1.0e6 ); // time step
  prevT = currT;
  int pos = 0; 
  
    //MOTOR 2
  // set target position
  int target2 = 1800;
 
  // PID constants
  float kp2 = 1;
  float kd2 = 0.0;
  float ki2 =0.0;
 
  long currT2 = micros();
  float deltaT2 = ((float) (currT2 - prevT2))/( 1.0e6 ); // time step
  prevT2 = currT2;
  int pos2 = 0;

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pos = posi,
    pos2 = posi2;
  }
  int e = pos - target; // error
  float dedt = (e-eprev)/(deltaT); // derivative
  float eintegral = eintegral + e*deltaT; // integral
  float u = kp*e + kd*dedt + ki*eintegral;

  // motor power
  float pwr = fabs(u);
  if( pwr > 255 ){
    pwr = 255;
  }
  // motor direction
  int dir = 1;
  if(u<0){
    dir = -1;
  }
  // signal the motor
  setMotor(dir,pwr,PWM,IN1,IN2);
  // store previous error
  eprev = e;
  Serial.print(target);
  Serial.print(" ");
  Serial.print(pos);
  Serial.print("motor1");
  Serial.println();

  // error2
  int e2 = pos2 - target2;
  float dedt2 = (e2-eprev2)/(deltaT2); // derivative 2
  float eintegral2 = eintegral2 + e2*deltaT2; // integral 2
  float u2 = kp2*e2 + kd2*dedt2 + ki2*eintegral2; // control loop signal 2
  float pwr2 = fabs(u2);// motor power
  if( pwr2 > 255 ){
    pwr2 = 255;
  }
  // motor direction
  int dir2 = 1;
  if(u2<0){
    dir2 = -1;
  }
  // signal the motor
  setMotor(dir2,pwr2,PWM2,IN3,IN4);
  // store previous error
  eprev2 = e2;
  Serial.print(target2);
  Serial.print(" ");
  Serial.print(pos2);
  Serial.print(e2);

  Serial.print("motor2");
  Serial.println();
}







//Motor 1
void setMotor(int dir, int pwmVal, int pwm, int in1, int in2){
  analogWrite(pwm,pwmVal);
  if(dir == 1){
    digitalWrite(in1,HIGH);
    digitalWrite(in2,LOW);
  }
  else if(dir == -1){
    digitalWrite(in1,LOW);
    digitalWrite(in2,HIGH);
  }
  else{
    digitalWrite(in1,LOW);
    digitalWrite(in2,LOW);
  }  
}

//Motor 2
void setMotor2(int dir2, int pwmVal2, int pwm2, int in3, int in4){
  analogWrite(pwm2,pwmVal2);
  if(dir2 == 1){
    digitalWrite(in3,HIGH);
    digitalWrite(in4,LOW);
  }
  else if(dir2 == -1){
    digitalWrite(in3,LOW);
    digitalWrite(in4,HIGH);
  }
  else{
    digitalWrite(in3,LOW);
    digitalWrite(in4,LOW);
  }  
}

//Motor 1
void readEncoder(){
  int b = digitalRead(ENCB);
  if(b > 0){
    posi++;
  }
  else{
    posi--;
  }
}
//Motor 2
void readEncoder2(){
  int b2 = digitalRead(ENCD);
  if(b2 > 0){
    posi2++;
  }
  else{
    posi2--;
  }
}
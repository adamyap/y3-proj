#include <util/atomic.h> // For the ATOMIC_BLOCK macro

//Motor 1
#define ENCA 3 // YELLOW from encoder(interrupt pin)
#define ENCB 9 // WHITE from encoder
#define PWM 5
#define IN2 6
#define IN1 7

//Motor 2
#define ENCC 2 // YELLOW from encoder(interrupt pin)
#define ENCD 8 // WHITE from encoder
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

////////////////////////////////////////////////////////////////////////////////////////////

//Runs Once
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
  
}

/////////////////////////////////////////////////////////////////////////////////////////

//Runs continuously 
void loop() {

  //MOTOR 1
  int pos = 0; 
  int target = 2100; // MOTOR 1 target positon
  long currT = micros(); // time instance
  float deltaT = ((float) (currT - prevT))/( 1.0e6 ); // time step
  prevT = currT;
 
  // PID constants (MOTOR 1)
  float kp = 1;
  float kd = 0.0;
  float ki = 0.0;
   
  //MOTOR 2
  int pos2 = 0;
  int target2 = 1800; // MOTOR 2 target postion
  long currT2 = micros();
  float deltaT2 = ((float) (currT2 - prevT2))/( 1.0e6 ); // time step
  prevT2 = currT2;

  // PID constants (MOTOR 2)
  float kp2 = 1;
  float kd2 = 0.0;
  float ki2 =0.0;
  
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pos = posi,
    pos2 = posi2;
  }

  // MOTOR 1 FEEDBACK LOOP
  int e = pos - target; // error 1
  float dedt = (e-eprev)/(deltaT); // derivative 1
  float eintegral = eintegral + e*deltaT; // integral 1
  float u = kp*e + kd*dedt + ki*eintegral; // control loop 1

  // MOTOR 1 Power
  float pwr = fabs(u);
  if( pwr > 255 ){
    pwr = 255;
  }

  // MOTOR 1 Direction
  int dir = 1;
  if(u<0){
    dir = -1;
  }

  // Drive MOTOR 1
  setMotor(dir,pwr,PWM,IN1,IN2);
  eprev = e; // store previous error

  // MOTOR 1 POSITION MONITORING
  Serial.print(target);
  Serial.print(" ");
  Serial.print(pos);
  Serial.print("motor1");
  Serial.println();

  // MOTOR 2 FEEDBACK LOOP
  int e2 = pos2 - target2; // error 2
  float dedt2 = (e2-eprev2)/(deltaT2); // derivative 2
  float eintegral2 = eintegral2 + e2*deltaT2; // integral 2
  float u2 = kp2*e2 + kd2*dedt2 + ki2*eintegral2; // control loop signal 2

  // MOTOR 2 Power
  float pwr2 = fabs(u2);
  if( pwr2 > 255 ){
    pwr2 = 255;
  }
  // MOTOR 2 Drection
  int dir2 = 1;
  if(u2<0){
    dir2 = -1;
  }
  // MOTOR 2 Drive
  setMotor(dir2,pwr2,PWM2,IN3,IN4);
  eprev2 = e2;// store previous error

  //MOTOR 2 POSITION MONITORING/ debugging
  Serial.print(target2);
  Serial.print(" ");
  Serial.print(pos2);
  Serial.print(e2);
  Serial.print("motor2");
  Serial.println();
}

///////////////////////////////////////////////////////////////////////////////

// Motor drive function
void setMotor(int direction, int pwmValue, int pwmPIN, int inAPIN, int inBPIN){
  analogWrite(pwmPIN,pwmValue);
  if(direction == 1){
    digitalWrite(inAPIN,HIGH);
    digitalWrite(inBPIN,LOW);
  }
  else if(direction == -1){
    digitalWrite(inAPIN,LOW);
    digitalWrite(inBPIN,HIGH);
  }
  else{
    digitalWrite(inAPIN,LOW);
    digitalWrite(inBPIN,LOW);
  }  
}

//Motor 1 (interupt triggered on rising encoder A)
void readEncoder(){
  int b = digitalRead(ENCB);
  if(b > 0){
    posi++;
  }
  else{
    posi--;
  }
}
//Motor 2 (interupt triggered on rising encoder C)
void readEncoder2(){
  int d = digitalRead(ENCD);
  if(d > 0){
    posi2++;
  }
  else{
    posi2--;
  }
}

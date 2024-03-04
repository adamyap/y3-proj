#include <util/atomic.h> // For the ATOMIC_BLOCK macro

//Motor A
#define ENCA 3 // YELLOW from encoder (interrupt pin)
#define ENCB 9 // WHITE from encoder for monitoring encoder level
#define PWM_A 5
#define IN2 7
#define IN1 6

//Motor B
#define ENCC 2 // YELLOW from encoder (interrupt pin)
#define ENCD 8 // WHITE from encoder, for monitoring encoder level
#define PWM_B 10
#define IN4 11
#define IN3 12

//Motor A
volatile int posi_A = 0; // specify posi_A as volatile: https://www.arduino.cc/reference/en/language/variables/variable-scope-qualifiers/volatile/
volatile int desiredPositionA;
long prevT_A = 0;
float eprev_A = 0;
float eintegral_A = 0;

//Motor B
volatile int posi_B = 0; // specify posi_B as volatile: https://www.arduino.cc/reference/en/language/variables/variable-scope-qualifiers/volatile/
volatile int desiredPositionB;
long prevT_B = 0;
float eprev_B = 0;
float eintegral_B = 0;

////////////////////////////////////////////////////////////////////////////////////////////

//Runs Once
void setup() {
  Serial.begin(9600);
  //Motor 1
  pinMode(ENCA,INPUT);
  pinMode(ENCB,INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA),readEncoderA_A,RISING);
  pinMode(PWM_A,OUTPUT);
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);

  //Motor 2
  pinMode(ENCC,INPUT);
  pinMode(ENCD,INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCC),readEncoderC_B,RISING);
  pinMode(PWM_B,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);
  
}

/////////////////////////////////////////////////////////////////////////////////////////

//Runs continuously 
void loop() {

// Communication with Python Via serial line
  if (Serial.available()) {
    char motor = Serial.read(); // Read which motor to move (expect 'A' or 'B')
    if(motor == 'A') {
      desiredPositionA = Serial.parseInt(); // Read desired position for motor A
    } 
    else if(motor == 'B') {
      desiredPositionB = Serial.parseInt(); // Read desired position for motor B
    }
  }

  //MOTOR A
  int pos_A = 0; 
  int target_A = desiredPositionA; // MOTOR 1 target positon
  long currT_A = micros(); // time instance
  float deltaT_A = ((float) (currT_A - prevT_A))/( 1.0e6 ); // time step
  prevT_A = currT_A;
   
  //MOTOR B
  int pos_B = 0;
  int target_B = desiredPositionB; // MOTOR 2 target postion
  long currT_B = micros();
  float deltaT_B = ((float) (currT_B - prevT_B))/( 1.0e6 ); // time step
  prevT_B = currT_B;

  // PID constants (MOTOR A)
  float kp_A = 1.2;
  float kd_A = 0.11;
  float ki_A = 0;

  // PID constants (MOTOR B)
  float kp_B = 1.2;
  float kd_B = 0.1;
  float ki_B = 0;
  
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pos_A = posi_A,
    pos_B = posi_B;
  }

  // MOTOR A FEEDBACK LOOP
  int e_A = pos_A - target_A; // error 1
  float de_Adt = (e_A-eprev_A)/(deltaT_A); // derivative 1
  float e_Aintegral = eintegral_A + e_A*deltaT_A; // integral 1
  float u_A = kp_A*e_A + kd_A*de_Adt + ki_A*e_Aintegral; // control loop 1

  // MOTOR A Power
  float pwr_A = fabs(u_A);
  if( pwr_A > 255 ){
    pwr_A = 255;
  }

  // MOTOR A Direction
  int dir_A = 1;
  if(u_A<0){
    dir_A = -1;
  }

  // Drive MOTOR A
  setMotor(dir_A,pwr_A,PWM_A,IN1,IN2);
  eprev_A = e_A; // store previous error

  // MOTOR A POSITION MONITORING
  //Serial.print(target_A);
  //Serial.print(" ");
  //Serial.print(pos);
  //Serial.print("motor_A");
  //Serial.println();

  // MOTOR B FEEDBACK LOOP
  int e_B = pos_B - target_B; // error 2
  float de_Bdt = (e_B-eprev_B)/(deltaT_B); // derivative 2
  float e_Bintegral = e_Bintegral + e_B*deltaT_B; // integral 2
  float u_B = kp_B*e_B + kd_B*de_Bdt + ki_B*e_Bintegral; // control loop signal 2

  // MOTOR B Power
  float pwr_B = fabs(u_B);
  if( pwr_B > 255 ){
    pwr_B = 255;
  }
  // MOTOR B Drection
  int dir_B = 1;
  if(u_B<0){
    dir_B = -1;
  }
  // MOTOR B Drive
  setMotor(dir_B,pwr_B,PWM_B,IN3,IN4);
  eprev_B = e_B;// store previous error

  //MOTOR B POSITION MONITORING/ debugging
  //Serial.print(target_B);
  //Serial.print(" ");
  //Serial.print(pos2);
  //Serial.print(e2);
  //Serial.print("motor_B");
  //Serial.println();
}

///////////////////////////////////////////////////////////////////////////////

// Motor Drive function
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

//Motor 1 (interupt triggered on rising encoder A form Motor A)
void readEncoderA_A(){
  int b = digitalRead(ENCB);
  if(b > 0){
    posi_A++;
  }
  else{
    posi_A--;
  }
}
//Motor 2 (interupt triggered on rising encoder C from Motor B)
void readEncoderC_B(){
  int d = digitalRead(ENCD);
  if(d > 0){
    posi_B++;
  }
  else{
    posi_B--;
  }
}

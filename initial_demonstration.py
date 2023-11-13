import serial
import time

ser = serial.Serial('COM11', 9600)  # replace 'COM11' with the port where your Arduino is connected
time.sleep(2)  # wait for the serial connection to initialize

def move_servos(angle1, angle2):
    if 0 <= angle1 <= 100 and 0 <= angle2 <= 100:
        ser.write(f"{angle1},{angle2}".encode())
        time.sleep(1)  # wait for the servos to move

while True:
    angle1 = int(input("Enter an angle for servo1 (0-100): "))
    angle2 = int(input("Enter an angle for servo2 (0-100): "))
    move_servos(angle1, angle2)
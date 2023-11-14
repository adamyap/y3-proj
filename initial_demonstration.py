import serial
import time

ser = serial.Serial('COM11', 9600)  # replace 'COM11' with the port where your Arduino is connected
time.sleep(2)  # wait for the serial connection to initialize

def move_servos(angle1, angle2):
    if 0 <= angle1 <= 120 and 0 <= angle2 <= 120:
        ser.write(f"{angle1},{angle2}".encode())
        #time.sleep(1)  # wait for the servos to move

def continuous_rotation():
    move_servos(120, 120)
    time.sleep(1)  # adjust this delay for speed control
    move_servos(0, 0)
    time.sleep(1)

def simple_maze():
    move_servos(60, 60)
    time.sleep(2)
    move_servos(0, 0)
    time.sleep(1.5)
    move_servos(120, 120)
    time.sleep(1)
    move_servos(120, 0)
    time.sleep(1)
    move_servos(0, 120)
    time.sleep(1)
    move_servos(0, 0)
    time.sleep(1)
    move_servos(120, 0) ####
    time.sleep(0.23)
    move_servos(0, 30)
    time.sleep(2)
    move_servos(120, 120)
    time.sleep(1)
    move_servos(120, 0)
    time.sleep(1)
    move_servos(120, 120)
    time.sleep(1)
    move_servos(0, 50)
    time.sleep(1)
    move_servos(0, 120)
    time.sleep(1)
    move_servos(120, 120)
    time.sleep(1)
    move_servos(30, 120)
    time.sleep(1)
    move_servos(120, 120)
    time.sleep(1)
    move_servos(120, 0)
    time.sleep(1)
    move_servos(0, 0)
    time.sleep(1)   
    move_servos(120, 0)
    time.sleep(1)
    move_servos(0, 0)
    time.sleep(1)
    move_servos(120, 0)
    time.sleep(1)
    move_servos(0, 0)
    time.sleep(1)
    move_servos(0, 120)
    time.sleep(1)
    move_servos(0, 0)
    time.sleep(1)
    move_servos(60, 60)


while True:
    #angle1 = int(input("Enter an angle for servo1 (0-120): "))
    #angle2 = int(input("Enter an angle for servo2 (0-120): "))
    #move_servos(angle1, angle2)
    #continuous_rotation()
    simple_maze()
    break

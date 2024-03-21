import serial
import time

ser = serial.Serial('COM3',9600)

def move_servos(angle1, angle2):
    if -60 <= angle1 <= 60 and -60 <= angle2 <= 60:
        angle1 = angle1 + 60
        angle2 = angle2 + 60
        ser.write(f"x,{angle1}".encode())
        ser.write(f"y,{angle2}".encode())
        print(f"Sent command: {angle1},{angle2}")

def continuous_rotation():
    move_servos(10,0)
    time.sleep(0.05)
    move_servos(-10,-0)
    time.sleep(0.05)

def simple_maze():
    move_servos(60, 60)
    time.sleep(3)
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
    time.sleep(0.3)
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

move_servos(0, 0)
time.sleep(2)

while True:
    #angle1 = int(input("Enter angle for servo1 (0-120): "))
    #angle2 = int(input("Enter angle for servo2 (0-120): "))
    
    continuous_rotation()
    #simple_maze()

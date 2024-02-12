import serial
import time
from tkinter import Tk, Scale, HORIZONTAL

ser = serial.Serial('COM11', 9600)  # replace 'COM11' with the port where your Arduino is connected
time.sleep(2)  # wait for the serial connection to initialize

def move_servos(angle1, angle2):
    if 0 <= angle1 <= 100 and 0 <= angle2 <= 100:
        ser.write(f"{angle1},{angle2}".encode())

def update_servos(_=None):  # the argument is needed as Scale's command callback passes the new value to the function
    angle1 = servo1_scale.get()
    angle2 = servo2_scale.get()
    move_servos(angle1, angle2)

root = Tk()

servo1_scale = Scale(root, from_=0, to=100, length=400, orient=HORIZONTAL, command=update_servos)
servo1_scale.pack()

servo2_scale = Scale(root, from_=0, to=100, length=400, orient=HORIZONTAL, command=update_servos)
servo2_scale.pack()

root.mainloop()

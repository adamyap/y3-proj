import serial
import time
import math
import random

# Initialize serial connection
ser = serial.Serial('COM3', 9600)
time.sleep(2)  # Wait for the connection to establish (avoid initialization bits)

startup_angle = 10
scale = 20

def calculate_rotor_angles(ball_coordx1, ball_coordy1, ball_coordx2, ball_coordy2, velocity):
    distancex = ball_coordx2 - ball_coordx1  # east/west for rotor 1 (longer edge)
    distancey = ball_coordy2 - ball_coordy1  # south/north for rotor 2 (shorter edge)
    
    if distancex != 0:
        angle = math.atan(distancey / distancex)  # angle between x axis and movement vector
    else:
        angle = math.pi / 2 if distancey > 0 else -math.pi / 2
    
    if distancex > 0:
        Ranglex = startup_angle + velocity * scale * abs(math.cos(angle))
    elif distancex == 0:
        Ranglex = 0
    else:
        Ranglex = -startup_angle - velocity * scale * abs(math.cos(angle))
    
    if distancey > 0:
        Rangley = startup_angle + velocity * scale * abs(math.sin(angle))
    elif distancey == 0:
        Rangley = 0
    else:
        Rangley = -startup_angle - velocity * scale * abs(math.sin(angle))

    # Limiting Ranglex and Rangley between -60 and 60
    Ranglex = max(min(Ranglex, 60), -60)
    Rangley = max(min(Rangley, 60), -60)

    return round(2100/450*Ranglex), round(2100/360*Rangley)

def send_position(motor, position):
    """
    Sends a motor position command to the Arduino.
    :param motor: 'A' or 'B', indicating which motor to control
    :param position: The desired position as an integer
    """
    command = f"{motor}{position}\n"  # Format the command string
    ser.write(command.encode())  # Encode and send the command
    print(f"Sent command: {command}")

if __name__ == "__main__":
    try:
        while True:
            send_position('A', 200)
            send_position('B', -200)
            time.sleep(0.2)
            send_position('A', -200)
            send_position('B', 200)
            
            time.sleep(0.2)  # Adjust delay as necessary
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        ser.close()  # Ensure the serial connection is closed properly

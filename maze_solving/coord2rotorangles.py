import math
import time
import random

startup_angle = 10
scale = 20

def calculate_rotor_angles(ball_coordx1, ball_coordy1, ball_coordx2, ball_coordy2, velocity):
    distancex = ball_coordx2 - ball_coordx1 # east/west for rotor 1(longer edge)
    distancey = ball_coordy2 - ball_coordy1 # south/north for rotor 2(shorter edge)
    if distancex != 0:
        angle = math.atan(distancey / distancex) # angle between x axis and movement vector
    else:
        if distancey > 0: 
            angle = math.pi/2
        else:
            angle = -math.pi/2
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

while True:
    ball_coordx1 = random.uniform(0, 1080)
    ball_coordy1 = random.uniform(0, 720)
    ball_coordx2 = random.uniform(0, 1080)
    ball_coordy2 = random.uniform(0, 720)
    velocity = random.uniform(1, 5)
    
    result = calculate_rotor_angles(ball_coordx1, ball_coordy1, ball_coordx2, ball_coordy2, velocity)
    print("A ", result[0])
    print("B ", result[1])
    
    time.sleep(1)  # 100ms delay

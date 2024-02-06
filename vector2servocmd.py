import math

startup_angle = 10
scale = 20

ball_coordx1 = int(input("insert ball coordx1:"))
ball_coordy1 = int(input("insert ball coordy1:"))
ball_coordx2 = int(input("insert ball coordx2:"))
ball_coordy2 = int(input("insert ball coordy2:"))
velocity = int(input("insert moving speed in px/s:"))

distancex = ball_coordx2 - ball_coordx1  # east/west for rotor 1(longer edge)
distancey = ball_coordy2 - ball_coordy1  # south/north for rotor 2(shorter edge)
angle = math.atan(distancey / distancex)

if distancex > 0:
    Ranglex = startup_angle + velocity * scale * (abs(math.cos(angle)))
elif distancex == 0:
    Ranglex = 0
else:
    Ranglex = -startup_angle - velocity * scale * (abs(math.cos(angle)))

if distancey > 0:
    Rangley = startup_angle + velocity * scale * (abs(math.sin(angle)))
elif distancey == 0:
    Rangley = 0
else:
    Rangley = -startup_angle - velocity * scale * (abs(math.sin(angle)))

# Limiting Ranglex and Rangley between -60 and 60
Ranglex = max(min(Ranglex, 60), -60)
Rangley = max(min(Rangley, 60), -60)

print("angle =:", math.degrees(angle))
print("rotorx angle =:", round(Ranglex, 2), "rotory angle =:", round(Rangley, 2))

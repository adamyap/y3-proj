This is the place for control software.

Currently:

Arduino code works by reading inputs from the serial port.
- In the form of "angle1,angle2", where angle1 refers to servo1 and likewise.
- This is achieved by a python line:
    ser.write(f"{angle1},{angle2}".encode())
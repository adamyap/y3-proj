import serial
import time

# change 'COM3' as needed
ser = serial.Serial('COM4', 9600)
time.sleep(2)  # Wait for the connection to establish (avoid initilisation bits)

def send_position(motor, position):
    """
    Sends a motor position command to the Arduino.

    :param motor: 'A' or 'B', indicating which motor to control
    :param position: The desired position as an integer
    """
    command = f"{motor}{position}\n"  # Format the command string
    ser.write(command.encode())  # Encode and send the command
    print(f"Sent command: {command}")

#Create an infinte loop which monitors for user input
while True:
    # User input prompt
    user_input = input("Enter command (Motor Position) or 'exit': ")

    # Use 'exit' to stop communicating with Arduino
    if user_input.lower() == 'exit':
        break

    try:
        # Splitting the input into motor and position
        motor, position = user_input.split()
        position = int(position)  # Convert position to integer

        # Validate motor input
        if motor.upper() in ['A', 'B']:
            send_position(motor.upper(), position)
        else:
            print("Invalid motor. Please enter 'A' or 'B' followed by the position.")
    except ValueError:
        print("Invalid input format. Please enter the command in the format 'Motor Position'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    time.sleep(0.1)  # Short delay to ensure command is processed

# Close the serial connection
ser.close()
print("Program terminated.")

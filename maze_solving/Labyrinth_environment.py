import gym
from gym import spaces
import numpy as np
import cv2  # OpenCV for camera operations


class BrioLabyrinthEnv(gym.Env):
    """Custom OpenAI.gym Environment for the BRIO Labyrinth game controlled by DC motors with Encoders."""
    def __init__(self):
        super(BrioLabyrinthEnv, self).__init__()
        # Initialisation of environmen properties (NEEDS WORK)
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.current_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.ball_position = None  # track the ball position
        # Add plate angle and velocity detection 
        

        self.camera = self.initialize_camera()
        self.motor_controller = self.initialize_motor_controller()
        # Piezometer / Automatic Ball deployer down the line?

        self.goal_area = ((x1, y1), (x2, y2))  # Define the goal area in terms of pixel coordinates (identify the coloured end region from rectified image)
        self.hole_positions = [...]  # Define positions of holes if relevant (from adams detection)
        
        # additional contraints to be added as needed

def step(self, action):
    # Apply action to the motors
    # is this done by the nural network actor?
    self.apply_action_to_motors(action)
    
    # Capture the next state from the camera
    # purely just camera image 
    next_state = self.capture_image()
    
    # Calculate reward based on the new state
    # labyrinth must be divided into check/way points for progression to be rewarded
    # negative reward for cheating (possibly implement reward for sequential waypoints)
    reward = self.calculate_reward(next_state)
    
    # Check if ball has reached desitination goal
    done = self.is_done(next_state)
    
    # return additional info
    info = {}
    
    return next_state, reward, done, info

def reset(self):
    # Reset the labyrinth and motors to the initial state (0,0 on each motor)
    self.reset_labyrinth()
    
    # Capture the initial state from the camera
    initial_state = self.capture_image()
    
    return initial_state

def render(self, mode='human'):
    if mode == 'human':
        # Show the image or any other debug info
        cv2.imshow("BRIO Labyrinth", self.current_image)
        cv2.waitKey(1)

def close(self):
    

def capture_image(self):
    # Camera observation code...
    # 1. Plate angle from reference landmarks
    # 2. Possible velocity vector

    return image

def apply_action_to_motors(self, action):
    # Motor control linking with arduino here

    
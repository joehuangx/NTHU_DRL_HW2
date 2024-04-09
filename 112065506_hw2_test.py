import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from collections import deque


class model(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(model, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # Flatten the output for fully connected layers
        self.flatten = nn.Flatten()
        # Define the fully connected layers
        self.linear = nn.Linear(3136, 512)
        self.fc = nn.Linear(512, num_actions)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        return self.fc(x)


class Agent:
    def __init__(self):
        # Initialize observation dimension and action dimension
        self.obs_dim = 4
        self.action_dim = 12
        # Create an instance of the model
        self.Q = model(self.obs_dim, self.action_dim)
        # Path to the model file
        self.model_path = "112065506_hw2_data"

        # Load the model
        self.Q.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        # Initialize frame count and number of frames to stack
        self.frame_count = 0
        self.num_stack = 4
        # Initialize deque to store frames
        self.frames = deque(maxlen=self.num_stack)
        # Define the shape of the frame
        self.shape = (84, 84)
        # Initialize the last action taken
        self.last_action = 1

    def GrayScale(self, observation):
        # Convert RGB observation to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

    def Resize(self, observation):
        # Resize the observation to the specified shape
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation

    def preprocess(self, observation):
        # Preprocess the observation
        observation = self.Resize(observation)
        observation = self.GrayScale(observation)
        return observation

    def act(self, obs):
        # Preprocess the observation
        obs = self.preprocess(obs)

        # Add the observation to frames
        self.frames.append(obs)

        # When there are four observations in frames, combine them into one frame
        if len(self.frames) == 4:
            obs_stack = np.stack(self.frames, axis=0)
            obs_stack = np.expand_dims(obs_stack, axis=0)
            obs_stack = torch.tensor(obs_stack, dtype=torch.float32)

            # Clear frames
            self.frames = []

            # Use the frame for prediction
            action = self.Q(obs_stack)
            self.last_action = np.argmax(action.detach().numpy())

        return self.last_action

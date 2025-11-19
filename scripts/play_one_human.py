"""
Example of how to use Gym env, with one human player and one agent.

Humans can override controls for the LEFT Agent:

W - Jump
A - Left
D - Right

The RIGHT Agent is always controlled by the AI.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from time import sleep
import pygame
import slimevolley

# game settings:
RENDER_MODE = True

if __name__=="__main__":

  human_action = [0, 0, 0] # Action for the LEFT agent (human)
  HUMAN_PLAYER_ACTIVE = True 

  policy = slimevolley.BaselinePolicy() # defaults to use RNN Baseline for player

  env = slimevolley.SlimeVolleyEnv()
  env.seed(np.random.randint(0, 10000))

  if RENDER_MODE:
    env.render()

  obs, info = env.reset()

  steps = 0
  total_reward = 0
  
  done = False

  while not done:

    if RENDER_MODE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                # Controls for the LEFT agent (human)
                if event.key == pygame.K_d: human_action[0] = 1 # right
                if event.key == pygame.K_a: human_action[1] = 1 # left
                if event.key == pygame.K_w: human_action[2] = 1 # Jump
            if event.type == pygame.KEYUP:
                # Controls for the LEFT agent (human)
                if event.key == pygame.K_d: human_action[0] = 0
                if event.key == pygame.K_a: human_action[1] = 0
                if event.key == pygame.K_w: human_action[2] = 0

    # Action for the RIGHT agent (AI controlled)
    ai_action = policy.predict(obs)
    
    # Step the environment with ai_action for the right agent and human_action for the left agent
    obs, reward, terminated, truncated, info = env.step(ai_action, human_action)
    
    done = terminated or truncated

    if reward > 0 or reward < 0:
      print("reward", reward)

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.01)

    # make the game go slower for human players to be fair to humans.
    if HUMAN_PLAYER_ACTIVE:
      if slimevolley.PIXEL_MODE:
        sleep(0.01)
      else:
        sleep(0.02)

  env.close()
  print("cumulative score", total_reward)

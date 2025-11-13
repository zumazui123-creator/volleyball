"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gymnasium as gym
import slimevolleygym
import pygame

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True


if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  blue Agent:
  W - Jump
  A - Left
  D - Right

  Yellow Agent:
  Up Arrow, Left Arrow, Right Arrow
  """

  if RENDER_MODE:
    from time import sleep

  manualAction = [0, 0, 0] # forward, backward, jump
  otherManualAction = [0, 0, 0]
  manualMode = False
  otherManualMode = False

  policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

  env = gym.make("SlimeVolley-v0")
  obs, info = env.reset(seed=np.random.randint(0, 10000))

  if RENDER_MODE:
    env.render()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False

  while not done:

    if RENDER_MODE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: manualAction[0] = 1
                if event.key == pygame.K_RIGHT: manualAction[1] = 1
                if event.key == pygame.K_UP: manualAction[2] = 1
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP): manualMode = True
                if event.key == pygame.K_d: otherManualAction[0] = 1
                if event.key == pygame.K_a: otherManualAction[1] = 1
                if event.key == pygame.K_w: otherManualAction[2] = 1
                if event.key in (pygame.K_d, pygame.K_a, pygame.K_w): otherManualMode = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT: manualAction[0] = 0
                if event.key == pygame.K_RIGHT: manualAction[1] = 0
                if event.key == pygame.K_UP: manualAction[2] = 0
                if event.key == pygame.K_d: otherManualAction[0] = 0
                if event.key == pygame.K_a: otherManualAction[1] = 0
                if event.key == pygame.K_w: otherManualAction[2] = 0

    if manualMode: # override with keyboard
      action = manualAction
    else:
      action = policy.predict(obs)

    if otherManualMode:
      otherAction = otherManualAction
      obs, reward, terminated, truncated, info = env.step(action, otherAction)
    else:
      obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

    if reward > 0 or reward < 0:
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.02) # 0.01

  env.close()
  print("cumulative score", total_reward)

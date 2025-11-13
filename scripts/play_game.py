"""
Example of how to use Gym env, in single or multiplayer setting

Humans can override controls:

left Agent:
W - Jump
A - Left
D - Right

right Agent:
Up Arrow, Left Arrow, Right Arrow
"""

import numpy as np
from time import sleep
import pygame
import slimevolleygym

# game settings:
RENDER_MODE = True

if __name__=="__main__":

  manualAction = [0, 0, 0] # forward, backward, jump
  otherManualAction = [0, 0, 0]
  manualMode = False
  otherManualMode = False

  policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

  env = slimevolleygym.SlimeVolleyEnv()
  env.seed(np.random.randint(0, 10000))

  if RENDER_MODE:
    env.render()

  obs, info = env.reset()

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
      print("reward", reward)
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.01)

    # make the game go slower for human players to be fair to humans.
    if (manualMode or otherManualMode):
      if slimevolleygym.PIXEL_MODE:
        sleep(0.01)
      else:
        sleep(0.02)

  env.close()
  print("cumulative score", total_reward)

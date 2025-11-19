"""
Human vs AI in pixel observation environment

Note that for multiagent mode, otherObs's image is horizontally flipped

Performance, 100,000 frames in 144.839 seconds, or 690 fps.
"""

import gymnasium as gym
import slimevolleygym
from time import sleep
import pygame
import cv2

if __name__=="__main__":

  manualAction = [0, 0, 0] # forward, backward, jump
  manualMode = False

  # Pygame setup
  pygame.init()
  screen_width = 2160 // 2
  screen_height = 1080 // 2
  screen = pygame.display.set_mode((screen_width, screen_height))
  pygame.display.set_caption("Slime Volley Pixel")

  env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")

  policy = slimevolleygym.BaselinePolicy() # throw in a default policy (based on state, not pixels)

  obs = env.reset()
  
  done = False
  while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  manualAction[0] = 1
            if event.key == pygame.K_RIGHT: manualAction[1] = 1
            if event.key == pygame.K_UP:    manualAction[2] = 1
            if (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP): manualMode = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:  manualAction[0] = 0
            if event.key == pygame.K_RIGHT: manualAction[1] = 0
            if event.key == pygame.K_UP:    manualAction[2] = 0

    if manualMode: # override with keyboard
      action = manualAction # now just work w/ multibinary if it is not scalar
    else:
      state = info['state'] # cheat and look at the actual state (to find default actions quickly)
      action = policy.predict(state)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    img = env.render()
    img = cv2.resize(img, (screen_width, screen_height))
    surf = pygame.surfarray.make_surface(img)
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    
    sleep(0.02)
    
    if done:
      obs = env.reset()

  pygame.quit()
  env.close()

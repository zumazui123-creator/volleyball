"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import cv2 # installed with gym anyways
from collections import deque
import pygame
from .config import *

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# by default, don't load rendering (since we want to use it in headless cloud machines)
def setPixelObsMode():
  """
  used for experimental pixel-observation mode
  note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims (will be downsampled)

  also, both agent colors are identical, to potentially facilitate multiagent
  """
  global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
  PIXEL_MODE = True
  WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
  WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
  FACTOR = WINDOW_WIDTH / REF_W
  AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
  AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR

def upsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH * PIXEL_SCALE, PIXEL_HEIGHT * PIXEL_SCALE), interpolation=cv2.INTER_NEAREST)
def downsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT), interpolation=cv2.INTER_AREA)



from .game import Game

from .agent import Agent

from .policy import BaselinePolicy

from .config import *



class SlimeVolleyEnv(gym.Env):
  """
  Gym wrapper for Slime Volley game.

  By default, the agent you are training controls the right agent
  on the right. The agent on the left is controlled by the baseline
  RNN policy.

  Game ends when an agent loses 5 matches (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  # for compatibility with typical atari wrappers
  atari_action_meaning = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
  }
  atari_action_set = {
    0, # NOOP
    4, # LEFT
    7, # UPLEFT
    2, # UP
    6, # UPRIGHT
    3, # RIGHT
  }

  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)

  from_pixels = False
  atari_mode = False
  survival_bonus = False # Depreciated: augment reward, easier to train
  multiagent = True # optional args anyways

  def __init__(self, **kwargs):
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py

    Setting self.from_pixels to True makes the observation with multiples
    of 84, since usual atari wrappers downsample to 84x84
    """
    self.t = 0
    self.t_limit = 3000

    #self.action_space = spaces.Box(0, 1.0, shape=(3,))
    if self.atari_mode:
      self.action_space = spaces.Discrete(6)
    else:
      self.action_space = spaces.MultiBinary(3)

    if self.from_pixels:
      setPixelObsMode()
      self.observation_space = spaces.Box(low=0, high=255,
        shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
    else:
      high = np.array([np.finfo(np.float32).max] * 12)
      self.observation_space = spaces.Box(-high, high)
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game()
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function

    self.policy = BaselinePolicy() # the “bad guy”

    self.viewer = None
    self.screen = None
    self.clock = None

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(np_random=self.np_random)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]

  def getObs(self):
    if self.from_pixels:
      obs = self.render(mode='state')
      self.canvas = obs
    else:
      obs = self.game.agent_right.getObservation()
    return obs

  def discreteToBox(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 3:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 6)
    return self.action_table[n]

  def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1

    if self.otherAction is not None:
      otherAction = self.otherAction
      
    if otherAction is None: # override baseline policy
      obs = self.game.agent_left.getObservation()
      otherAction = self.policy.predict(obs)

    if self.atari_mode:
      action = self.discreteToBox(action)
      otherAction = self.discreteToBox(otherAction)

    self.game.agent_left.setAction(otherAction)
    self.game.agent_right.setAction(action) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    terminated = False
    truncated = False
    if self.t >= self.t_limit:
      truncated = True
    if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
      terminated = True

    otherObs = None
    if self.multiagent:
      if self.from_pixels:
        otherObs = cv2.flip(obs, 1) # horizontal flip
      else:
        otherObs = self.game.agent_left.getObservation()

    info = {
      'ale.lives': self.game.agent_right.lives(),
      'ale.otherLives': self.game.agent_left.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_right.getObservation(),
      'otherState': self.game.agent_left.getObservation(),
    }

    if self.survival_bonus:
      return obs, reward+0.01, terminated, truncated, info
    return obs, reward, terminated, truncated, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()

  def reset(self, **kwargs):
    self.init_game_state()
    return self.getObs(), {}

  def render(self, mode='human', close=False):
    if self.screen is None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

    canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    canvas.fill(BACKGROUND_COLOR)

    self.game.display(canvas)

    if mode == 'human':
        self.screen.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata['video.frames_per_second'])
    else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

  def close(self):
    if self.screen is not None:
      pygame.quit()
    
  def get_action_meanings(self):
    return [self.atari_action_meaning[i] for i in self.atari_action_set]

class SlimeVolleyPixelEnv(SlimeVolleyEnv):
  from_pixels = True

class SlimeVolleyAtariEnv(SlimeVolleyEnv):
  from_pixels = True
  atari_mode = True

class SlimeVolleySurvivalAtariEnv(SlimeVolleyEnv):
  from_pixels = True
  atari_mode = True
  survival_bonus = True

class SurvivalRewardEnv(gym.RewardWrapper):
  def __init__(self, env):
    """
    adds 0.01 to the reward for every timestep agent survives

    :param env: (Gym Environment) the environment
    """
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """
    adds that extra survival bonus for living a bit longer!

    :param reward: (float)
    """
    return reward + 0.01

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.

    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers

    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    return np.concatenate(list(self.frames), axis=2)

#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1

    if render_mode:
      env.render()

  return total_reward, t

def render_atari(obs):
  """
  Helper function that takes in a processed obs (84,84,4)
  Useful for visualizing what an Atari agent actually *sees*
  Outputs in Atari visual format (Top: resized to orig dimensions, buttom: 4 frames)
  """
  tempObs = []
  obs = np.copy(obs)
  for i in range(4):
    if i == 3:
      latest = np.copy(obs[:, :, i])
    if i > 0: # insert vertical lines
      obs[:, 0, i] = 141
    tempObs.append(obs[:, :, i])
  latest = np.expand_dims(latest, axis=2)
  latest = np.concatenate([latest*255.0] * 3, axis=2).astype(np.uint8)
  latest = cv2.resize(latest, (84 * 8, 84 * 4), interpolation=cv2.INTER_NEAREST)
  tempObs = np.concatenate(tempObs, axis=1)
  tempObs = np.expand_dims(tempObs, axis=2)
  tempObs = np.concatenate([tempObs*255.0] * 3, axis=2).astype(np.uint8)
  tempObs = cv2.resize(tempObs, (84 * 8, 84 * 2), interpolation=cv2.INTER_NEAREST)
  return np.concatenate([latest, tempObs], axis=0)

####################
# Reg envs for gym #
####################

register(
    id='SlimeVolley-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyEnv'
)

register(
    id='SlimeVolleyPixel-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyPixelEnv'
)

register(
    id='SlimeVolleyNoFrameskip-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyAtariEnv'
)

register(
    id='SlimeVolleySurvivalNoFrameskip-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleySurvivalAtariEnv'
)

if __name__=="__main__":
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

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  manualAction = [0, 0, 0] # forward, backward, jump
  otherManualAction = [0, 0, 0]
  manualMode = False
  otherManualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.UP:    manualAction[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

    if k == key.D:     otherManualAction[0] = 1
    if k == key.A:     otherManualAction[1] = 1
    if k == key.W:     otherManualAction[2] = 1
    if (k == key.D or k == key.A or k == key.W): otherManualMode = True

  def key_release(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.UP:    manualAction[2] = 0
    if k == key.D:     otherManualAction[0] = 0
    if k == key.A:     otherManualAction[1] = 0
    if k == key.W:     otherManualAction[2] = 0

  policy = BaselinePolicy() # defaults to use RNN Baseline for player

  env = SlimeVolleyEnv()
  env.seed(np.random.randint(0, 10000))
  #env.seed(721)

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False

  while not done:

    if manualMode: # override with keyboard
      action = manualAction
    else:
      action = policy.predict(obs)

    if otherManualMode:
      otherAction = otherManualAction
      obs, reward, done, _ = env.step(action, otherAction)
    else:
      obs, reward, done, _ = env.step(action)

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
      if PIXEL_MODE:
        sleep(0.01)
      else:
        sleep(0.02)

  env.close()
  print("cumulative score", total_reward)

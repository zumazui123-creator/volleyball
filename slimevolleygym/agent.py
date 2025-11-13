import math
import numpy as np
from .config import *

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.vx = 0
    self.vy = 0
    # ball
    self.bx = 0
    self.by = 0
    self.bvx = 0
    self.bvy = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.ovx = 0
    self.ovy = 0
  def getObservation(self):
    result = [self.x, self.y, self.vx, self.vy,
              self.bx, self.by, self.bvx, self.bvy,
              self.ox, self.oy, self.ovx, self.ovy]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir, x, y, c):
    self.dir = dir # -1 means left, 1 means right player for symmetry.
    self.x = x
    self.y = y
    self.r = 1.5
    self.c = c
    self.vx = 0
    self.vy = 0
    self.desired_vx = 0
    self.desired_vy = 0
    self.state = RelativeState()
    self.emotion = "happy"; # hehe...
    self.life = MAXLIVES
  def lives(self):
    return self.life
  def setAction(self, action):
    forward = False
    backward = False
    jump = False
    if action[0] > 0:
      forward = True
    if action[1] > 0:
      backward = True
    if action[2] > 0:
      jump = True
    self.desired_vx = 0
    self.desired_vy = 0
    if (forward and (not backward)):
      self.desired_vx = -PLAYER_SPEED_X
    if (backward and (not forward)):
      self.desired_vx = PLAYER_SPEED_X
    if jump:
      self.desired_vy = PLAYER_SPEED_Y
  def move(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def step(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def update(self):
    self.vy += GRAVITY * TIMESTEP

    if (self.y <= REF_U + NUDGE*TIMESTEP):
      self.vy = self.desired_vy

    self.vx = self.desired_vx*self.dir

    self.move()

    if (self.y <= REF_U):
      self.y = REF_U;
      self.vy = 0;

    # stay in their own half:
    if (self.x*self.dir <= (REF_WALL_WIDTH/2+self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_WALL_WIDTH/2+self.r)

    if (self.x*self.dir >= (REF_W/2-self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_W/2-self.r)
  def updateState(self, ball, opponent):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x*self.dir
    self.state.y = self.y
    self.state.vx = self.vx*self.dir
    self.state.vy = self.vy
    # ball
    self.state.bx = ball.x*self.dir
    self.state.by = ball.y
    self.state.bvx = ball.vx*self.dir
    self.state.bvy = ball.vy
    # opponent
    self.state.ox = opponent.x*(-self.dir)
    self.state.oy = opponent.y
    self.state.ovx = opponent.vx*(-self.dir)
    self.state.ovy = opponent.vy
  def getObservation(self):
    return self.state.getObservation()

  def display(self, canvas, bx, by):
    x = self.x
    y = self.y
    r = self.r

    angle = math.pi * 60 / 180
    if self.dir == 1:
      angle = math.pi * 120 / 180
    eyeX = 0
    eyeY = 0

    canvas = half_circle(canvas, toX(x), toY(y), toP(r), color=self.c)

    # track ball with eyes (replace with observed info later):
    c = math.cos(angle)
    s = math.sin(angle)
    ballX = bx-(x+(0.6)*r*c);
    ballY = by-(y+(0.6)*r*s);

    if (self.emotion == "sad"):
      ballX = -self.dir
      ballY = -3

    dist = math.sqrt(ballX*ballX+ballY*ballY)
    eyeX = ballX/dist
    eyeY = ballY/dist

    canvas = circle(canvas, toX(x+(0.6)*r*c), toY(y+(0.6)*r*s), toP(r)*0.3, color=(255, 255, 255))
    canvas = circle(canvas, toX(x+(0.6)*r*c+eyeX*0.15*r), toY(y+(0.6)*r*s+eyeY*0.15*r), toP(r)*0.1, color=(0, 0, 0))

    # draw coins (lives) left
    for i in range(1, self.life):
      canvas = circle(canvas, toX(self.dir*(REF_W/2+0.5-i*2.)), WINDOW_HEIGHT-toY(1.5), toP(0.5), color=COIN_COLOR)

    return canvas

import math
import numpy as np
from .config import *
from .config import half_circle
import pygame.draw

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self) -> None:
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
  def getObservation(self) -> np.ndarray:
    result = [self.x, self.y, self.vx, self.vy,
              self.bx, self.by, self.bvx, self.bvy,
              self.ox, self.oy, self.ovx, self.ovy]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir: int, x: float, y: float, c: tuple) -> None:
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
  def lives(self) -> int:
    return self.life
  def setAction(self, action: list) -> None:
    forward, backward, jump = action[0] > 0, action[1] > 0, action[2] > 0
    
    self.desired_vx = 0
    if forward and not backward:
        self.desired_vx = -PLAYER_SPEED_X
    elif backward and not forward:
        self.desired_vx = PLAYER_SPEED_X

    self.desired_vy = PLAYER_SPEED_Y if jump else 0
  def move(self) -> None:
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def update(self) -> None:
    self._apply_gravity()
    self._update_velocity()
    self.move()
    self._handle_collisions()

  def _apply_gravity(self) -> None:
    self.vy += GRAVITY * TIMESTEP

  def _update_velocity(self) -> None:
    if self.y <= REF_U + NUDGE * TIMESTEP:
        self.vy = self.desired_vy
    self.vx = self.desired_vx * self.dir

  def _handle_collisions(self) -> None:
    # ground collision
    if self.y <= REF_U:
        self.y = REF_U
        self.vy = 0

    # wall collisions
    if self.x * self.dir <= (REF_WALL_WIDTH / 2 + self.r):
        self.vx = 0
        self.x = self.dir * (REF_WALL_WIDTH / 2 + self.r)

    if self.x * self.dir >= (REF_W / 2 - self.r):
        self.vx = 0
        self.x = self.dir * (REF_W / 2 - self.r)
  def updateState(self, ball: "Particle", opponent: "Agent") -> None:
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
  def getObservation(self) -> np.ndarray:
    return self.state.getObservation()

  def display(self, env, canvas: "pygame.Surface", bx: float, by: float) -> "pygame.Surface":
    self._draw_body(env, canvas)
    self._draw_eyes(env, canvas, bx, by)
    self._draw_lives(env, canvas)
    return canvas

  def _draw_body(self, env, canvas: "pygame.Surface") -> None:
    half_circle(canvas, env.toX(self.x), env.toY(self.y), env.toP(self.r), color=self.c, dir=self.dir)

  def _draw_eyes(self, env, canvas: "pygame.Surface", bx: float, by: float) -> None:
    angle = math.pi * RIGHT_AGENT_ANGLE / 180
    if self.dir == 1:
        angle = math.pi * LEFT_AGENT_ANGLE / 180

    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    eye_x_offset = self.x + EYE_OFFSET_X_FACTOR * self.r * cos_angle
    eye_y_offset = self.y + EYE_OFFSET_X_FACTOR * self.r * sin_angle

    ball_x_rel = bx - eye_x_offset
    ball_y_rel = by - eye_y_offset

    if self.emotion == "sad":
        ball_x_rel = -self.dir * SAD_EMOTION_BALL_X
        ball_y_rel = SAD_EMOTION_BALL_Y

    dist = math.sqrt(ball_x_rel**2 + ball_y_rel**2)
    pupil_x_offset = ball_x_rel / dist
    pupil_y_offset = ball_y_rel / dist

    # eye
    pygame.draw.circle(canvas, (255, 255, 255),
                       (int(env.toX(eye_x_offset)), int(env.toY(eye_y_offset))),
                       int(env.toP(self.r * EYE_RADIUS_FACTOR)))
    # pupil
    pygame.draw.circle(canvas, (0, 0, 0),
                       (int(env.toX(eye_x_offset + pupil_x_offset * PUPIL_OFFSET_FACTOR * self.r)),
                        int(env.toY(eye_y_offset + pupil_y_offset * PUPIL_OFFSET_FACTOR * self.r))),
                       int(env.toP(self.r * PUPIL_RADIUS_FACTOR)))

  def _draw_lives(self, env, canvas: "pygame.Surface") -> None:
    for i in range(1, self.life):
        x_pos = self.dir * (REF_W / 2 + LIVES_OFFSET_X - i * LIVES_SPACING)
        y_pos = LIVES_OFFSET_Y
        pygame.draw.circle(canvas, COIN_COLOR,
                           (int(env.toX(x_pos)), int(env.toY(y_pos))),
                           int(env.toP(LIVES_RADIUS)))

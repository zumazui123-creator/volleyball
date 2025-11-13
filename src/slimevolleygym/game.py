import math
import numpy as np
from .config import *
from .agent import Agent
import pygame

class DelayScreen:
  """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """
  def __init__(self, life=INIT_DELAY_FRAMES):
    self.life = 0
    self.reset(life)
  def reset(self, life=INIT_DELAY_FRAMES):
    self.life = life
  def status(self):
    if (self.life == 0):
      return True
    self.life -= 1
    return False

class Particle:
  """ used for the ball, and also for the round stub above the fence """
  def __init__(self, x, y, vx, vy, r, c):
    self.x = x
    self.y = y
    self.prev_x = self.x
    self.prev_y = self.y
    self.vx = vx
    self.vy = vy
    self.r = r
    self.c = c
  def display(self, env, canvas):
    pygame.draw.circle(canvas, self.c, (int(env.toX(self.x)), int(env.toY(self.y))), int(env.toP(self.r)))
    return canvas
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def applyAcceleration(self, ax, ay):
    self.vx += ax * TIMESTEP
    self.vy += ay * TIMESTEP
  def checkEdges(self):
    self._check_horizontal_bounds()
    score = self._check_vertical_bounds()
    if score != NO_SCORE:
        return score
    self._check_fence_collision()
    return NO_SCORE

  def _check_horizontal_bounds(self):
    if self.x <= (self.r - REF_W / 2):
        self.vx *= -FRICTION
        self.x = self.r - REF_W / 2 + NUDGE * TIMESTEP

    if self.x >= (REF_W / 2 - self.r):
        self.vx *= -FRICTION
        self.x = REF_W / 2 - self.r - NUDGE * TIMESTEP

  def _check_vertical_bounds(self):
    if self.y <= (self.r + REF_U):
        self.vy *= -FRICTION
        self.y = self.r + REF_U + NUDGE * TIMESTEP
        if self.x <= 0:
            return BALL_SCORE_LEFT
        else:
            return BALL_SCORE_RIGHT
    if self.y >= (REF_H - self.r):
        self.vy *= -FRICTION
        self.y = REF_H - self.r - NUDGE * TIMESTEP
    return NO_SCORE

  def _check_fence_collision(self):
    if (self.x <= (REF_WALL_WIDTH / 2 + self.r)) and (self.prev_x > (REF_WALL_WIDTH / 2 + self.r)) and (self.y <= REF_WALL_HEIGHT):
        self.vx *= -FRICTION
        self.x = REF_WALL_WIDTH / 2 + self.r + NUDGE * TIMESTEP

    if (self.x >= (-REF_WALL_WIDTH / 2 - self.r)) and (self.prev_x < (-REF_WALL_WIDTH / 2 - self.r)) and (self.y <= REF_WALL_HEIGHT):
        self.vx *= -FRICTION
        self.x = -REF_WALL_WIDTH / 2 - self.r - NUDGE * TIMESTEP

  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    return (dx*dx+dy*dy)
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    return (r*r > self.getDist2(p)) # if distance is less than total radius, then colliding.
  def bounce(self, other):
    """
    Handles the physics of a collision between two particles.
    """
    # vector from other to self
    delta_x = self.x - other.x
    delta_y = self.y - other.y
    distance = math.sqrt(delta_x**2 + delta_y**2)

    # normalized vector
    normal_x = delta_x / distance
    normal_y = delta_y / distance

    # move particles apart to avoid overlap
    nudge_x = normal_x * NUDGE
    nudge_y = normal_y * NUDGE
    while self.isColliding(other):
        self.x += nudge_x
        self.y += nudge_y

    # relative velocity
    relative_vx = self.vx - other.vx
    relative_vy = self.vy - other.vy

    # dot product of relative velocity and normal vector
    dot_product = relative_vx * normal_x + relative_vy * normal_y

    # impulse vector
    impulse_x = normal_x * (dot_product * 2.0)
    impulse_y = normal_y * (dot_product * 2.0)

    # update velocities
    self.vx = (relative_vx - impulse_x) + other.vx
    self.vy = (relative_vy - impulse_y) + other.vy
  def limitSpeed(self, minSpeed, maxSpeed):
    mag2 = self.vx*self.vx+self.vy*self.vy;
    if (mag2 > (maxSpeed*maxSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= maxSpeed
      self.vy *= maxSpeed

    if (mag2 < (minSpeed*minSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= minSpeed
      self.vy *= minSpeed

class Wall:
  """ used for the fence, and also the ground """
  def __init__(self, x, y, w, h, c):
    self.x = x;
    self.y = y;
    self.w = w;
    self.h = h;
    self.c = c
  def display(self, env, canvas):
    pygame.draw.rect(canvas, self.c, (int(env.toX(self.x-self.w/2)), int(env.toY(self.y+self.h/2)), int(env.toP(self.w)), int(env.toP(self.h))))
    return canvas

class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, np_random=np.random):
    self.ball = None
    self.ground = None
    self.fence = None
    self.fenceStub = None
    self.agent_left = None
    self.agent_right = None
    self.delayScreen = None
    self.np_random = np_random
    self.reset()
  def _create_ball(self):
    ball_vx = self.np_random.uniform(low=-20, high=20)
    ball_vy = self.np_random.uniform(low=10, high=25)
    return Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5, c=BALL_COLOR)

  def reset(self):
    self.ground = Wall(0, 0.75, REF_W, REF_U, c=GROUND_COLOR)
    self.fence = Wall(0, 0.75 + REF_WALL_HEIGHT / 2, REF_WALL_WIDTH, (REF_WALL_HEIGHT - 1.5), c=FENCE_COLOR)
    self.fenceStub = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2, c=FENCE_COLOR)
    self.ball = self._create_ball()
    self.agent_left = Agent(-1, -REF_W / 4, 1.5, c=AGENT_LEFT_COLOR)
    self.agent_right = Agent(1, REF_W / 4, 1.5, c=AGENT_RIGHT_COLOR)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)
    self.delayScreen = DelayScreen()

  def newMatch(self):
    self.ball = self._create_ball()
    self.delayScreen.reset()
  def step(self):
    """ main game loop """
    self.betweenGameControl()
    self._update_agents()
    self._update_ball()
    self._handle_collisions()
    
    score = -self.ball.checkEdges()
    
    if score != NO_SCORE:
        self._handle_scoring(score)
        return score

    self._update_agent_states()
    return NO_SCORE

  def _update_agents(self):
    self.agent_left.update()
    self.agent_right.update()

  def _update_ball(self):
    if self.delayScreen.status():
        self.ball.applyAcceleration(0, GRAVITY)
        self.ball.limitSpeed(0, MAX_BALL_SPEED)
        self.ball.move()

  def _handle_collisions(self):
    if self.ball.isColliding(self.agent_left):
        self.ball.bounce(self.agent_left)
    if self.ball.isColliding(self.agent_right):
        self.ball.bounce(self.agent_right)
    if self.ball.isColliding(self.fenceStub):
        self.ball.bounce(self.fenceStub)

  def _handle_scoring(self, score):
    self.newMatch()
    if score < 0:  # baseline agent won
        self.agent_left.emotion = "happy"
        self.agent_right.emotion = "sad"
        self.agent_right.life -= 1
    else:
        self.agent_left.emotion = "sad"
        self.agent_right.emotion = "happy"
        self.agent_left.life -= 1
        
  def _update_agent_states(self):
    # update internal states (the last thing to do)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)

  def display(self, env, canvas):
    # background color
    # if PIXEL_MODE is True, canvas is an RGB array.
    # if PIXEL_MODE is False, canvas is viewer object
    # canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
    canvas = self.fence.display(env, canvas)
    canvas = self.fenceStub.display(env, canvas)
    canvas = self.agent_left.display(env, canvas, self.ball.x, self.ball.y)
    canvas = self.agent_right.display(env, canvas, self.ball.x, self.ball.y)
    canvas = self.ball.display(env, canvas)
    canvas = self.ground.display(env, canvas)
    return canvas
  def betweenGameControl(self):
    agent = [self.agent_left, self.agent_right]
    if (self.delayScreen.life > 0):
      pass
      '''
      for i in range(2):
        if (agent[i].emotion == "sad"):
          agent[i].setAction([0, 0, 0]) # nothing
      '''
    else:
      agent[0].emotion = "happy"
      agent[1].emotion = "happy"

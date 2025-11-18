import math
import pygame
# import pygame.gfxdraw # Removed as filled_pie is not available

from collections import namedtuple

# game settings:
RENDER_MODE = True

GameConfig = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise', 'rnn_mode'])

REF_W = 24*2
REF_H = REF_W
REF_U = 1.5 # ground height
REF_WALL_WIDTH = 1.0 # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10*1.75
PLAYER_SPEED_Y = 10*1.35
MAX_BALL_SPEED = 15*1.5
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5

MAXLIVES = 5 # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = False 
PIXEL_SCALE = 4 # first render at multiple of Pixel Obs resolution, then downscale. Looks better.

PIXEL_WIDTH = 84*2*1
PIXEL_HEIGHT = 84*1

# Colors
BALL_COLOR = (217, 79, 0)
AGENT_LEFT_COLOR = (35, 93, 188)
AGENT_RIGHT_COLOR = (255, 236, 0)
PIXEL_AGENT_LEFT_COLOR = (255, 191, 0) # AMBER
PIXEL_AGENT_RIGHT_COLOR = (255, 191, 0) # AMBER

BACKGROUND_COLOR = (11, 16, 19)
FENCE_COLOR = (102, 56, 35)
COIN_COLOR = FENCE_COLOR
GROUND_COLOR = (116, 114, 117)

# Game constants
BALL_SCORE_LEFT = -1
BALL_SCORE_RIGHT = 1
NO_SCORE = 0

# Agent display constants
EYE_OFFSET_X_FACTOR = 0.6
EYE_RADIUS_FACTOR = 0.3
PUPIL_RADIUS_FACTOR = 0.1
PUPIL_OFFSET_FACTOR = 0.15
LEFT_AGENT_ANGLE = 120
RIGHT_AGENT_ANGLE = 60
SAD_EMOTION_BALL_X = -1
SAD_EMOTION_BALL_Y = -3
LIVES_OFFSET_X = 0.5
LIVES_OFFSET_Y = 1.5
LIVES_SPACING = 2.0
LIVES_RADIUS = 0.5

# Policy constants
ACTION_THRESHOLD = 0.75



def half_circle(surface, x, y, r, color, dir):
    # Draw a full circle
    pygame.draw.circle(surface, color, (int(x), int(y)), int(r))

    # Draw a rectangle over half of the circle with the background color
    if dir == -1: # Left agent, now facing down
        # The rectangle will cover the top half
        rect_x = x - r
        rect_y = y - r
        rect_width = 2 * r
        rect_height = r
    else: # Right agent, now facing up
        # The rectangle will cover the bottom half
        rect_x = x - r
        rect_y = y
        rect_width = 2 * r
        rect_height = r
    
    pygame.draw.rect(surface, BACKGROUND_COLOR, (int(rect_x), int(rect_y), int(rect_width), int(rect_height)))
    return surface

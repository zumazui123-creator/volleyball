import numpy as np
import json
import os
from config import ACTION_THRESHOLD

class BaselinePolicy:
  """ Tiny RNN policy with only 120 parameters of otoro.net/slimevolley agent """
  def __init__(self):
    self.nGameInput = 8 # 8 states for agent
    self.nGameOutput = 3 # 3 buttons (forward, backward, jump)
    self.nRecurrentState = 4 # extra recurrent states for feedback.

    self.nOutput = self.nGameOutput + self.nRecurrentState
    self.nInput = self.nGameInput + self.nOutput
    
    # store current inputs and outputs
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)

    # Load weights and biases from JSON file
    model_path = os.path.join(os.path.dirname(__file__), 'assets', 'models', 'baseline_policy.json')
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    self.weight = np.array(model_data['weight'])
    self.bias = np.array(model_data['bias'])

    # unflatten weight, convert it into 7x15 matrix.
    self.weight = self.weight.reshape(self.nGameOutput + self.nRecurrentState,
      self.nGameInput + self.nGameOutput + self.nRecurrentState)
  def reset(self):
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)
  def _forward(self):
    self.prevOutputState = self.outputState
    self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)
  def _setInputState(self, obs):
    # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
    [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
    self.inputState[0:self.nGameInput] = np.array([x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy])
    self.inputState[self.nGameInput:] = self.outputState
  def _getAction(self):
    forward = 0
    backward = 0
    jump = 0
    if (self.outputState[0] > ACTION_THRESHOLD):
      forward = 1
    if (self.outputState[1] > ACTION_THRESHOLD):
      backward = 1
    if (self.outputState[2] > ACTION_THRESHOLD):
      jump = 1
    return [forward, backward, jump]
  def predict(self, obs):
    """ take obs, update rnn state, return action """
    self._setInputState(obs)
    self._forward()
    return self._getAction()

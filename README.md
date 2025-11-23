# Slime Volleyball Gym Environment

<p align="left">
  <img width="100%" src="https://otoro.net/img/slimegym/pixel.gif">
</p>

Slime Volleyball is a simple and addictive physics-based game. This repository provides a gym environment for single and multi-agent reinforcement learning experiments.

The agent's goal is to score by making the ball land on the opponent's ground. Each agent has five lives. The game ends when an agent loses all lives or after 3000 timesteps. The agent gets a reward of +1 for scoring and -1 when the opponent scores.

This environment is a Python port of the original [Neural Slime Volleyball](https://otoro.net/slimevolley/) JavaScript game, designed for fast and lightweight reinforcement learning research.

## Key Features
- **Lightweight:** Only requires `gym` and `numpy`.
- **Single and Multi-Agent:** Supports both single-player and multi-agent (self-play) scenarios.
- **Fast:** Runs at approximately 12,500 timesteps per second on a modern CPU for state-based observations.
- **Educational:** Includes a [tutorial](docs/TRAINING.md) on various training methods, suitable for learning RL concepts.
- **Pixel and State Observations:** Supports both state-vector and pixel-based observations for testing different types of RL agents.

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/hardmaru/slimevolleygym.git
cd slimevolleygym
pip install -e .
```

It is recommended to use a virtual environment to manage dependencies.

## How to Play

You can play the game against the built-in AI or another human player.

### Human vs. AI

To play as the left agent against the AI, run:

```bash
python scripts/play_one_human.py
```

**Controls:**
- **W**: Jump
- **A**: Move Left
- **D**: Move Right

### Human vs. Human

To play with another human, run:

```bash
python scripts/play_game.py
```

**Controls:**
- **Left Agent:** W, A, D
- **Right Agent:** Up, Left, Right Arrow Keys

## Project Structure

```
├───assets/         # Pre-trained models
├───docs/           # Documentation and figures
├───scripts/        # Scripts for playing, evaluation, and testing
├───training/       # Scripts for training agents
├───tests/          # Test files for the game logic
├───agent.py        # Agent class
├───config.py       # Configuration file
├───game.py         # Game logic
├───mlp.py          # Multi-Layer Perceptron model
├───policy.py       # Baseline policy
├───slimevolley.py  # Main environment file
└───utils.py        # Utility functions
```

## Training and Evaluation

The `training` directory contains scripts for training agents using various methods, including PPO, CMA-ES, and Genetic Algorithms. The `scripts/eval` directory contains scripts for evaluating trained agents against each other.

For a detailed guide on training and evaluation, please refer to the [TRAINING.md](docs/TRAINING.md) tutorial.

## Environments

The environment comes in two main flavors: state-based and pixel-based observations.

| Environment ID                | Observation Space | Action Space    |
| ----------------------------- | ----------------- | --------------|
| `SlimeVolley-v0`              | `Box(12)`         | `MultiBinary(3)` |
| `SlimeVolleyPixel-v0`         | `Box(84, 168, 3)` | `MultiBinary(3)` |
| `SlimeVolleyNoFrameskip-v0`   | `Box(84, 168, 3)` | `Discrete(6)`   |

### State-Space Observation

The 12-dimensional state vector represents the positions and velocities of the agents and the ball:

<img src="https://render.githubusercontent.com/render/math?math=\left(x_{agent}, y_{agent}, \dot{x}_{agent}, \dot{y}_{agent}, x_{ball}, y_{ball}, \dot{x}_{ball}, \dot{y}_{ball}, x_{opponent}, y_{opponent}, \dot{x}_{opponent}, \dot{y}_{opponent}\right)">

The origin (0, 0) is at the bottom of the fence.

## Citation

If you use this environment in your research, please cite it as follows:

```
@misc{slimevolleygym,
  author = {David Ha},
  title = {Slime Volleyball Gym Environment},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hardmaru/slimevolleygym}},
}
```
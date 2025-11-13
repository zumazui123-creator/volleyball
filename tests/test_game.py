import pytest
from src.slimevolleygym.game import Game

def test_game_creation():
    """
    Test that the Game object is created correctly.
    """
    game = Game()
    assert game is not None
    assert game.ball is not None
    assert game.agent_left is not None
    assert game.agent_right is not None
    assert game.ground is not None
    assert game.fence is not None
    assert game.fenceStub is not None
    assert game.delayScreen is not None

def test_game_step_after_delay():
    """
    Test that the game state changes after the initial delay.
    """
    game = Game()
    initial_ball_x = game.ball.x
    initial_ball_y = game.ball.y
    initial_agent_left_x = game.agent_left.x
    initial_agent_left_y = game.agent_left.y
    initial_agent_right_x = game.agent_right.x
    initial_agent_right_y = game.agent_right.y

    # Set a jump action for both agents
    game.agent_left.setAction([0, 0, 1])
    game.agent_right.setAction([0, 0, 1])

    for _ in range(game.delayScreen.life + 1):
        game.step()

    assert game.ball.x != initial_ball_x or game.ball.y != initial_ball_y
    assert game.agent_left.x != initial_agent_left_x or game.agent_left.y != initial_agent_left_y
    assert game.agent_right.x != initial_agent_right_x or game.agent_right.y != initial_agent_right_y

def test_scoring_left():
    """
    Test that the score is updated correctly when the left player scores.
    """
    game = Game()
    game.ball.y = game.ground.y - 1 # below the ground
    game.ball.x = -1 # on the left side
    score = game.step()
    assert score == 1 # right agent gets a point
    assert game.agent_left.life == 4
    assert game.agent_right.life == 5

def test_scoring_right():
    """
    Test that the score is updated correctly when the right player scores.
    """
    game = Game()
    game.ball.y = game.ground.y - 1 # below the ground
    game.ball.x = 1 # on the right side
    score = game.step()
    assert score == -1 # left agent gets a point
    assert game.agent_left.life == 5
    assert game.agent_right.life == 4
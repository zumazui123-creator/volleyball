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

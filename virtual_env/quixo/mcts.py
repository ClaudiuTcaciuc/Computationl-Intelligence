from copy import copy
import random
import numpy as np

from game import Player, Move, Game

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MCTSPlayer(Player):
    def __init__(self, iterations=1000):
        self.iterations = iterations

class MCTSNode:
    pass

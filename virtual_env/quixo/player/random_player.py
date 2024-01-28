import random
from game import Game, Move, Player
from utils import verifie_move

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        while True:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            slide = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            if verifie_move(from_pos, slide, game):
                return from_pos, slide
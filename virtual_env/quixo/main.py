import random
from game import Game, Move, Player

""" Strategy to implement:
    - Q-learning
    - Monte Carlo
    - Genetic Algorithm
"""

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class QLearningPlayer(Player):
    def __init__(self, alpha: float, gamma: float, epsilon: float) -> None:
        super().__init__()
        self._alpha = alpha # learning rate
        self._gamma = gamma # discount factor
        self._epsilon = epsilon # exploration rate
        self._q_table = {} # Q-table to store the Q-values for each state-action pair
    

if __name__ == '__main__':
    g = Game()
    g.print()
    player1 = MyPlayer()
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")

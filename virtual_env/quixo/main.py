import random

import numpy as np
from game import Game, Move, Player
from itertools import product

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
        """ Initialize the Q-learning player 
            - alpha: learning rate
            - gamma: discount factor
            - epsilon: exploration rate
            
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._q_table = {}  # Q-table to store the Q-values for each state-action pair
        self._epoch = 0
    
    def evolve_parameters(self) -> None:
        """ Evolve the parameters of the Q-learning player """
        pass
    
    def get_state(self, game: 'Game') -> tuple:
        """ Get the current state of the game """
        
        board_state = tuple(map(tuple, game.get_board()))
        current_player = game.get_current_player()
        return board_state, current_player
    
    def get_q_value(self, state: tuple, action: Move) -> float:
        """ Get the Q-value for a given state-action pair """
        return self._q_table.get((state, action), 0.0)
    
    def get_best_move(self, state: tuple) -> Move:
        """ Get the best move for a given state """
        possible_moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]
        best_move = max(possible_moves, key=lambda move: self.get_q_value(state, move))
        return best_move

    def choose_random_position(self, game:'Game') -> tuple[int, int]:
        """ Choose a random position on the board """
        return (random.randint(0, 4), random.randint(0, 4))
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """ Make a move """
        current_state = self.get_state(game)
        
        if random.uniform(0, 1) < self._epsilon:
            # Explore
            action = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        else:
            # Exploit
            action = self.get_best_move(current_state)
        
        from_pos = self.choose_random_position(game)
        return from_pos, action
    
    def update_q_value(self, state: tuple, action: Move, reward: float, next_state: tuple) -> None:
        """ Update the Q-value for a given state-action pair """
        current_q_value = self.get_q_value(state, action)
        max_next_q_value = max(self.get_q_value(next_state, next_action) for next_action in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        new_q_value = current_q_value + self._alpha * (reward + self._gamma * max_next_q_value - current_q_value)
        self._q_table[(state, action)] = new_q_value

def train_players(player1: Player, player2: Player, epochs: int = 2_000): # tried 10_000 but the results were the same
    """ Train the players against each other """
    print("Training players...")
    for epoch in range(epochs):
        game = Game()
        winner = game.play(player1, player2)
        
        reward = 1 if winner == 0 else -1
        
        if isinstance(player1, QLearningPlayer):
            player1.update_q_value(player1.get_state(game), player1.get_best_move(player1.get_state(game)), reward, player1.get_state(game))
        if isinstance(player2, QLearningPlayer):
            player2.update_q_value(player2.get_state(game), player2.get_best_move(player2.get_state(game)), (reward*-1), player2.get_state(game))
        
        if epoch % (1/10*epochs) == 0:
            print(f"percent complete: {epoch / epochs * 100}%")
            
    print("Training complete.")

def testing_players(player1: Player, player2: Player, epochs: int = 1_000):
    """ Test the Q-learning player against a random player or another Q-learning player """
    print("Testing Q-learning players...")
    player_1_wins = 0
    player_2_wins = 0

    # turn off exploration
    if isinstance(player1, QLearningPlayer):
        player1._epsilon = 0
    if isinstance(player2, QLearningPlayer):
        player2._epsilon = 0

    for _ in range(epochs):
        game = Game()
        winner = game.play(player1, player2)
        
        player_1_wins += 1 if winner == 0 else 0
        player_2_wins += 1 if winner == 1 else 0

    print(f"Player 1 wins: {player_1_wins}")
    print(f"Player 2 wins: {player_2_wins}")
    
    return player_1_wins/(player_1_wins+player_2_wins)

def Q_learning_strategy():
    """ Implement the Q-learning strategy """
    player1 = QLearningPlayer(0.6, 0.5, 0.3)
    player2 = QLearningPlayer(0.6, 0.5, 0.3)
    player3 = RandomPlayer()
    
    train_players(player1, player2)
    print("---- Q-learning 1 vs Q-learning 2 ----")
    testing_players(player1, player2)
    print("---- Q-learning 1 vs Random ----")
    testing_players(player1, player3)
    print("---- Q-learning 2 vs Random ----")
    testing_players(player2, player3)
    
    print("train a new q-learning player against a random player")
    player1 = QLearningPlayer(0.1, 0.9, 0.1)
    player2 = RandomPlayer()
    
    train_players(player1, player2)
    print("---- Q-learning 1 vs Random ----")
    testing_players(player1, player2)
    
def find_best_parameters():
    # 1 hour 30 minutes
    # best parameters: (0.6, 0.5, 0.3)
    alpha = np.linspace(0.1, 0.9, 9)
    gamma = np.linspace(0.1, 0.9, 9)
    epsilon = np.linspace(0.1, 0.9, 9)
    
    best_params = (0, 0, 0)
    best_wins = 0
    count = 0
    player3 = RandomPlayer()
    
    for (a, g, e) in product(alpha, gamma, epsilon):
        player1 = QLearningPlayer(a, g, e)
        train_players(player1, player3, epochs=1_000)
        wins = testing_players(player1, player3)
        if wins > best_wins:
            best_wins = wins
            best_params = (a, g, e)
        count += 1
        print(f"percent complete: {count / (len(alpha)*len(gamma)*len(epsilon)) * 100}%")
    
    print(f"Best parameters: {best_params}")

if __name__ == '__main__':
    Q_learning_strategy()
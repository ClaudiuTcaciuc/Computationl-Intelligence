from copy import deepcopy
import random

import numpy as np
from game import Game, Move, Player
from q_learning import QLearningPlayer
from mcts import MCTSPlayer
from itertools import product

""" Strategy to implement:
    - Q-learning -> done
        - best parameters: (0.6, 0.5, 0.3) found by computing train and test 
            accuracy for different values of alpha, gamma, and epsilon
        - used 2_000 epochs for training and 1_000 epochs for testing
        - trained by playing against another Q-learning player
        - uses a Q-table to store the Q-values for each state-action pair

    - Monte Carlo Tree Search
        - https://www.wikiwand.com/it/Ricerca_ad_albero_Monte_Carlo
        - https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
        - https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238#:~:text=In%20the%20selection%20phase%2C%20MCTS,values%20for%20each%20child%20node.
        
    - Genetic Algorithm
"""

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


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
    """ Test the player against another player """
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
    player1 = MCTSPlayer()
    player2 = RandomPlayer()
    
    game = Game()
    winner = game.play(player1, player2)
    print(winner)
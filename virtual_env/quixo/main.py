from game import Game
from player.genetic_player import GeneticPlayer
from player.random_player import RandomPlayer
from tqdm import tqdm
import time

""" Strategy to implement:
    - Q-learning -> done
        - best parameters: (0.6, 0.5, 0.3) found by computing train and test 
            accuracy for different values of alpha, gamma, and epsilon
        - used 2_000 epochs for training and 1_000 epochs for testing
        - trained by playing against another Q-learning player
        - uses a Q-table to store the Q-values for each state-action pair
        
    - Genetic Algorithm -> done 
    
    - Project need a refactoring
    - Write a .md file
"""


def play_game(args):
    player_1, player_2 = args
    game = Game()
    winner = game.play(player_1, player_2)
    return winner

def genetic_algorithm_strategy():
    player1 = GeneticPlayer()
    player2 = RandomPlayer()

    num_games = 100
    
    print("------ Genetic Algorithm (with memory) vs Random -----")
    args_list = [(player1, player2) for _ in range(num_games)]
    
    results = []
    for args in tqdm(args_list, total=num_games, desc="Games Played"):
        results.append(play_game(args))

    player1_wins = results.count(0)

    print(f"Genetic Player winrate: {(player1_wins / num_games * 100):.2f}% against Random Player\n")

    player3 = GeneticPlayer(memory=0)

    print("------ Genetic Algorithm (without memory) vs Random -----")
    args_list = [(player3, player2) for _ in range(num_games)]
    
    results = []
    for args in tqdm(args_list, total=num_games, desc="Games Played"):
        results.append(play_game(args))

    player1_wins = results.count(0)

    print(f"Genetic Player winrate: {(player1_wins / num_games * 100):.2f}% against Random Player\n")
    
    print("------ Genetic Algorithm (with memory) vs Genetic Algorithm (without memory) -----")
    args_list = [(player1, player3) for _ in range(num_games)]
    
    results = []
    for args in tqdm(args_list, total=num_games, desc="Games Played"):
        results.append(play_game(args))

    player1_wins = results.count(0)
    
    print(f"Genetic Player winrate: {(player1_wins / num_games * 100):.2f}% against Genetic Player (without memory)\n")

if __name__ == '__main__':
    time_start = time.time()
    genetic_algorithm_strategy()
    time_end = time.time()
    print(f"Time elapsed: {(time_end - time_start):.2f}")
from game import Game, Player
from player.genetic_player import GeneticPlayer
from player.random_player import RandomPlayer
from player.qlearn_player import QLearningPlayer
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
    
    reward = 100 if winner == 0 else -10
    if isinstance(player_1, QLearningPlayer):
        player_1.update_q_table(reward, game)
    
    return winner

def genetic_algorithm_strategy():
    player1 = GeneticPlayer()
    player2 = RandomPlayer()
    player3 = GeneticPlayer(memory=0)

    num_games = 100
    
    print("------ Genetic Algorithm (with memory) vs Random -----")
    args_list = [(player1, player2) for _ in range(num_games)]
    
    results = []
    for args in tqdm(args_list, total=num_games, desc="Games Played"):
        results.append(play_game(args))

    player1_wins = results.count(0)

    print(f"Genetic Player winrate: {(player1_wins / num_games * 100):.2f}% against Random Player\n")

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

def q_learn_strategy(player1: QLearningPlayer, player2: Player, epochs: int = 200, test: bool = False):
    num_games = epochs
    args_list = [(player1, player2) for _ in range(num_games)]
    
    if test and isinstance(player1, QLearningPlayer):
        player1.exploration_prob = 0
    if test and isinstance(player2, QLearningPlayer):
        player2.exploration_prob = 0
    
    results = []
    for args in tqdm(args_list, total=num_games, desc="Games Played"):
        results.append(play_game(args))

    player1_wins = results.count(0)
    
    # if isinstance(player1, QLearningPlayer):
    #     for entry in player1.q_table:
    #         print(entry, player1.q_table[entry])
        
    if test:
        print(f"Q-learning Player winrate: {(player1_wins / num_games * 100):.2f}% against Random Player\n")
    else:
        print("train done")

if __name__ == '__main__':
    # time_start = time.time()
    # genetic_algorithm_strategy()
    # time_end = time.time()
    # print(f"Time elapsed: {(time_end - time_start):.2f}")
    player1 = QLearningPlayer(0.3, 0.7, 0.3)
    player2 = RandomPlayer()
    # game = Game()
    # winner = game.play(player1, player2)
    # winner = "Q_learn" if winner == 0 else "Random"
    # print(winner)
    q_learn_strategy(player1, player2)
    q_learn_strategy(player1, player2, test=True, epochs=100)
    
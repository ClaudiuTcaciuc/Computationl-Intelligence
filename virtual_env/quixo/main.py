from game import Game, Player
from player.genetic_player import GeneticPlayer
from player.random_player import RandomPlayer
from player.qlearn_player import QLearningPlayer
from tqdm import tqdm
import time

def play_game(args):
    """ Play a single game of tic-tac-toe. """
    player_1, player_2 = args
    game = Game()
    winner = game.play(player_1, player_2)
    
    reward = 10 if winner == 0 else -1
    if isinstance(player_1, QLearningPlayer):
        player_1.update_q_table(reward, game)
    
    return winner

def play_games(player1: Player, player2: Player, num_epochs: int = 100):
    """ Play a number of games of tic-tac-toe. """
    args_list = [(player1, player2) for _ in range(num_epochs)]
    
    results = []
    for args in tqdm(args_list, total=num_epochs, desc="Playing games"):
        results.append(play_game(args))
        
    player1_wins = results.count(0)
    player2_wins = results.count(1)
    
    return player1_wins, player2_wins

def main():
    """ Main function. """
    start_time = time.time()
    player1 = QLearningPlayer()
    player2 = GeneticPlayer()
    player3 = RandomPlayer()
    player4 = GeneticPlayer(memory=0)
    
    # train QLearningPlayer against GeneticPlayer
    print(f"---- Training {player1.name()} against {player2.name()} ----")
    play_games(player1, player2, num_epochs=200)
    
    # test QLearningPlayer against GeneticPlayer
    print(f"---- Testing {player1.name()} against {player2.name()} ----")
    
    if isinstance(player1, QLearningPlayer):
        player1.exploration_prob = 0
    
    player1_wins, player2_wins = play_games(player1, player2, num_epochs=100)
    print(f"{player1.name()} wins: {player1_wins}")
    print(f"{player2.name()} wins: {player2_wins}\n")
    
    player1 = QLearningPlayer()
    # train QLearningPlayer against RandomPlayer
    print(f"---- Training {player1.name()} against {player3.name()} ----")
    play_games(player1, player3, num_epochs=200)
    
    # test QLearningPlayer against RandomPlayer
    if isinstance(player1, QLearningPlayer):
        player1.exploration_prob = 0
    print(f"---- Testing {player1.name()} against {player3.name()} ----")
    player1_wins, player3_wins = play_games(player1, player3, num_epochs=100)
    print(f"{player1.name()} wins: {player1_wins}")
    print(f"{player3.name()} wins: {player3_wins}\n")
    
    # GeneticPlayer against RandomPlayer
    print(f"---- Testing {player2.name()} against {player3.name()} ----")
    player2_wins, player3_wins = play_games(player2, player3, num_epochs=100)
    print(f"{player2.name()} wins: {player2_wins}")
    print(f"{player3.name()} wins: {player3_wins}\n")
    
    # GeneticPlayer against GeneticPlayer without memory
    print(f"---- Testing {player2.name()} against {player4.name()} ----")
    player2_wins, player4_wins = play_games(player2, player4, num_epochs=100)
    print(f"{player2.name()} wins: {player2_wins}")
    print(f"{player4.name()} wins: {player4_wins}\n")
    
    # GeneticPlayer without memory against RandomPlayer
    print(f"---- Testing {player4.name()} against {player3.name()} ----")
    player4_wins, player3_wins = play_games(player4, player3, num_epochs=100)
    print(f"{player4.name()} wins: {player4_wins}")
    print(f"{player3.name()} wins: {player3_wins}\n")
    
    end_time = time.time()
    print(f"---- Total execution time {end_time - start_time} seconds ----")


if __name__ == '__main__':
    main()
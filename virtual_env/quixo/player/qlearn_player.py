import numpy as np
from game import Player, Move, Game
from utils import verifie_move

class QLearningPlayer(Player):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}
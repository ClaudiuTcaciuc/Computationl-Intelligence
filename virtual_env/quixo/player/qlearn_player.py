import random
import numpy as np
from game import Player, Move, Game
from utils import verifie_move, fitness

class QLearningPlayer(Player):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}
        self.history = []
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_state = tuple(map(tuple, game.get_board()))
        
        if random.uniform(0, 1) < self.exploration_prob:
            # Explore
            from_pos, move = self.__explore(game)
        else:
            # Exploit
            from_pos, move = self.__exploit(current_state, game)
            
        self.history.append((current_state, from_pos, move))
        return from_pos, move
    
    def __explore(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """ Explore the environment """
        while True:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            if verifie_move(from_pos, move, game):
                return from_pos, move
    
    def __exploit(self, current_state: tuple, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_state = tuple(map(tuple, current_state))

        state = self.q_table.get(current_state)
        if not state:
            return self.__explore(game)
        
        return state["from_pos"], state["move"]

    def update_q_table(self, reward: float, game: Game) -> None:
        for state, from_pos, move in reversed(self.history):
            if state not in self.q_table:
                self.q_table[state] = {
                    "value": 0,
                    "from_pos": from_pos,
                    "move": move
                }
            else:
                if fitness(from_pos, move, game) > fitness(self.q_table[state]["from_pos"], self.q_table[state]["move"], game):
                    self.q_table[state]["from_pos"] = from_pos
                    self.q_table[state]["move"] = move
            max_value = max(self.q_table[state]["value"] for state in self.q_table)
            self.q_table[state]["value"] += self.learning_rate * (reward + self.discount_factor * max_value - self.q_table[state]["value"])

        self.history = []
        
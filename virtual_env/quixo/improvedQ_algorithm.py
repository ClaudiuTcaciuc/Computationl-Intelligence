import random
from game import Game, Move, Player

from utils import verifie_move, simulate_move, fitness


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
    
    def get_state(self, game: 'Game') -> tuple:
        """ Get the current state of the game """
        board_state = tuple(map(tuple, game.get_board()))
        current_player = game.get_current_player()
        return board_state, current_player
    
    def get_q_value(self, state: tuple, from_pos: tuple[int, int], move: Move) -> float:
        """ Get the Q-value for a given state-action pair """
        return self._q_table.get((state, from_pos, move), 0.0)
    
    def select_action(self, state: tuple, valid_moves: list[Move]) -> Move:
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        else:
            return max(valid_moves, key=lambda m: self.get_q_value(state, m), default=None)
    
    def generate_random_valid_action(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """ Generate a random valid move """
        while True:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            if verifie_move(from_pos, move, game):
                return from_pos, move

    
    def get_best_action(self, state: tuple) -> tuple:
        """ Get the best move for a given state """
        # search inside the Q-table for the best action for the given state
        state_entry = [entry for entry in self._q_table if entry[0] == state]
        if not state_entry:
            return None
        
        best_entry = max(state_entry, key=lambda entry: self.get_q_value(entry[0], entry[1], entry[2]))
        return best_entry
    
    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        current_state = self.get_state(game)
        
        if random.uniform(0, 1) < self._epsilon:
            # Explore
            from_pos, move = self.generate_random_valid_action(game)
        else:
            # Exploit based on sorted Q-values
            sorted_actions = sorted(self._q_table, key=lambda entry: self.get_q_value(entry[0], entry[1], entry[2]), reverse=True)
            
            best_action = None
            for action in sorted_actions:
                if action[0] == current_state and verifie_move(action[1], action[2], game):
                    best_action = action
                    break
            
            if best_action is None:
                from_pos, move = self.generate_random_valid_action(game)
            else:
                from_pos, move = best_action[1], best_action[2]

        # add state-action pair to the Q-table
        next_state = simulate_move(from_pos, move, game)
        reward = fitness(from_pos, move, game)
        self.update_q_value(current_state, from_pos, move, reward, (tuple(map(tuple, next_state)), game.get_current_player()))
        
        return from_pos, move
    
    def update_q_value(self, state: tuple, from_pos: tuple[int, int], move: Move, reward: float, next_state: tuple) -> None:
        """ Update the Q-value for a given state-action pair """
        current_q_value = self.get_q_value(state, from_pos, move)
        best_next_action = self.get_best_action(next_state)
        if best_next_action is None:
            next_q_value = 0.0
        else:
            next_q_value = self.get_q_value(next_state, best_next_action[1], best_next_action[2])
        
        new_q_value = current_q_value + self._alpha * (reward + self._gamma * next_q_value - current_q_value)
        self._q_table[(state, from_pos, move)] = new_q_value
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
        self._q_table = {}  # Q-table to store the Q-values for each state
        self._epoch = 0
    
    def evolve_parameters(self) -> None:
        """ Evolve the parameters of the Q-learning player 
            Placeholder - Customize based on your needs
        """
        # Your logic for parameter evolution goes here
        pass
    
    def get_state(self, game: 'Game') -> tuple:
        """ Get the current state of the game """
        board_state = tuple(map(tuple, game.get_board()))
        current_player = game.get_current_player()
        return board_state, current_player
    
    def get_q_value(self, state: tuple) -> tuple[Move, tuple[int, int], float]:
        """ Get the Q-value for a given state """
        return self._q_table.get(state, (None, None, 0.0))
    
    def get_best_move(self, state: tuple) -> tuple[Move, tuple[int, int]]:
        """ Get the best move for a given state """
        best_move, from_pos, _ = self.get_q_value(state)
        return best_move, from_pos

    def choose_random_position(self, game:'Game') -> tuple[int, int]:
        """ Choose a random position on the board """
        return (random.randint(0, 4), random.randint(0, 4))
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """ Make a move """
        current_state = self.get_state(game)
        
        if random.uniform(0, 1) < self._epsilon:
            # Explore
            action = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            from_pos = self.choose_random_position(game)
        else:
            # Exploit
            best_move, from_pos = self.get_best_move(current_state)
            action = best_move if best_move else random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            from_pos = from_pos if from_pos else self.choose_random_position(game)
        
        # Update Q-table
        next_state = current_state if not verifie_move(from_pos, action, game) else tuple(map(tuple, simulate_move(from_pos, action, game)))
        player_id = game.get_current_player()
        reward = 1
        if game.check_winner() == player_id:
            reward = 2
        else:
            reward = -1

        _, _, current_q_value = self.get_q_value(current_state)
        max_next_q_value = max(self.get_q_value(next_state)[2] for action in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        new_q_value = current_q_value + self._alpha * (reward + self._gamma * max_next_q_value - current_q_value)
        
        # Update Q-table with the new values
        self._q_table[current_state] = (action, from_pos, new_q_value)
        
        return from_pos, action

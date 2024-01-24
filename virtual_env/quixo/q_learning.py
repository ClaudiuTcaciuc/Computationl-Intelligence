import random
from game import Game, Move, Player


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
        """ Evolve the parameters of the Q-learning player 
            tried to implement but it didn't achive better results    
        """
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
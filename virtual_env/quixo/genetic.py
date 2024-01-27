from copy import deepcopy
import random
import numpy as np
from game import Player, Move, Game

class GeneticPlayer(Player):
    def __init__(self, population_size=100, generations=50, train=1) -> None:
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.best_moves = {}  # Dictionary to store the best moves for each board
        self.initial_population = []
        self.train = train
    
    def evolve_population(self, population: list[tuple[tuple[int, int], Move]], game: 'Game') -> list[tuple[tuple[int, int], Move]]:        
        sorted_population = sorted(population, key=lambda move: self.fitness(move, game), reverse=True)
        elite_size = int(self.population_size * 0.2)
        elite = sorted_population[:elite_size]
        
        offspring = []
        for _ in range(self.population_size - elite_size):
            parent_1, parent_2 = random.sample(elite, 2)
            crossover_point = random.randint(1, len(parent_1) - 1)
            child1 = parent_1[:crossover_point] + parent_2[crossover_point:]
            offspring.append(child1)
        
        mutated_offspring = [self.mutate(move) for move in offspring]
        return elite + mutated_offspring
    
    def mutate(self, move: tuple[tuple[int, int], Move]) -> tuple[tuple[int, int], Move]:
        mutation_rate = 0.1
        if random.uniform(0, 1) < mutation_rate:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            return (from_pos, move)
        return move
    
    def simulate_move(self, from_pos: tuple[int, int], slide: Move, game: 'Game') -> tuple[bool, np.array]:
        copy_game = deepcopy(game)
        copy_board = copy_game.get_board()
        
        if game.get_current_player() > 2:
            return False, copy_board
        
        prev_value = deepcopy(copy_board[(from_pos[1], from_pos[0])])
        acceptable = self.simulate_take((from_pos[1], from_pos[0]), copy_game)
        if acceptable:
            acceptable = self.simulate_slide((from_pos[1], from_pos[0]), slide, copy_game)
            if not acceptable:
                copy_board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable, copy_board
    
    def simulate_take(self, from_pos: tuple[int, int], game: 'Game') -> bool:
        acceptable: bool = (
            (from_pos[0] == 0 and from_pos[1] < 5)
            or (from_pos[0] == 4 and from_pos[1] < 5)
            or (from_pos[1] == 0 and from_pos[0] < 5)
            or (from_pos[1] == 4 and from_pos[0] < 5)
        ) and (game.get_board()[from_pos] < 0 or game.get_board()[from_pos] == game.get_current_player())
        if acceptable:
            game.get_board()[from_pos] = game.get_current_player()
        return acceptable
    
    def simulate_slide(self, from_pos: tuple[int, int], slide: Move, game: 'Game') -> bool:
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        
        if from_pos not in SIDES:
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        else:
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT
            )
            acceptable_bottom: bool = from_pos == (0, 4) and (
                slide == Move.TOP or slide == Move.RIGHT
            )
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.BOTTOM or slide == Move.LEFT
            )
            acceptable_right: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT
            )
        acceptable = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        
        if acceptable:
            piece = game.get_board()[from_pos]
            if slide == Move.LEFT:
                for i in range(from_pos[1], 0, -1):
                    game.get_board()[(from_pos[0], i)] = game.get_board()[(from_pos[0], i - 1)]
                game.get_board()[(from_pos[0], 0)] = piece
            elif slide == Move.RIGHT:
                for i in range(from_pos[1], 4):
                    game.get_board()[(from_pos[0], i)] = game.get_board()[(from_pos[0], i + 1)]
                game.get_board()[(from_pos[0], 4)] = piece
            elif slide == Move.TOP:
                for i in range(from_pos[0], 0, -1):
                    game.get_board()[(i, from_pos[1])] = game.get_board()[(i - 1, from_pos[1])]
                game.get_board()[(0, from_pos[1])] = piece
            elif slide == Move.BOTTOM:
                for i in range(from_pos[0], 4):
                    game.get_board()[(i, from_pos[1])] = game.get_board()[(i + 1, from_pos[1])]
                game.get_board()[(4, from_pos[1])] = piece
        return acceptable
    
    def fitness(self, move: tuple[tuple[int, int], Move], game: 'Game') -> int:
        board = game.get_board()
        
        player_symbol = game.get_current_player()
        board_array = np.array(board)

        # Count player symbols in rows, columns, and diagonals
        row_scores = np.sum(board_array == player_symbol, axis=1)
        col_scores = np.sum(board_array == player_symbol, axis=0)
        principal_diag_score = np.sum(np.diag(board_array) == player_symbol)
        secondary_diag_score = np.sum(np.diag(np.fliplr(board_array)) == player_symbol)

        # Calculate distances and penalties
        row_distance = np.sum(np.abs(row_scores - 5))
        col_distance = np.sum(np.abs(col_scores - 5))
        principal_diag_distance = np.abs(principal_diag_score - 5)
        secondary_diag_distance = np.abs(secondary_diag_score - 5)

        # # Penalty for non-contiguous pieces
        # contiguous_penalty = np.sum((board_array[:, 1:] == player_symbol) & (board_array[:, :-1] != player_symbol))

        # # Penalty for opponent's pieces in the same row, column, or diagonal
        # opponent_penalty = np.sum(np.any(board_array == player_symbol, axis=1) & np.any(board_array == game_copy.get_current_player(), axis=1))
        # opponent_penalty += 1 if board_array[2, 2] == player_symbol and game_copy.get_current_player() == board_array[2, 2] else 0

        # Combine scores and penalties
        tot_score = row_distance + col_distance + principal_diag_distance + secondary_diag_distance
        return tot_score
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = tuple(map(tuple, game.get_board()))
        
        for _ in range(self.population_size):
            while True:
                from_pos = (random.randint(0, 4), random.randint(0, 4))
                move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
                valid, _ = self.simulate_move((from_pos[1], from_pos[0]), move, game)
                if valid:
                    self.initial_population.append((from_pos, move))
                    break
        
        for _ in range(self.generations):
            self.initial_population = self.evolve_population(self.initial_population, game)
        sorted_population = sorted(self.initial_population, key=lambda move: self.fitness(move, game), reverse=True)
        best_move = sorted_population[0]
        current_best_fitness = self.fitness(best_move, game)
        print(f"best move: {best_move}, fitness: {current_best_fitness}, board: {current_board}, player: {game.get_current_player()}")
        if current_board in self.best_moves:
            previous_best_fitness = self.best_moves[current_board][1]
            if current_best_fitness >= previous_best_fitness:
                self.best_moves[current_board] = best_move, current_best_fitness
            else:
                return self.best_moves[current_board][0]
        else:
            self.best_moves[current_board] = best_move, current_best_fitness
        
        from_pos = best_move[0]
        move = best_move[1]
        return from_pos, move

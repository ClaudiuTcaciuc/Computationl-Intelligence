from copy import deepcopy
import random
import numpy as np
from game import Player, Move, Game


class GeneticPlayer(Player):
    def __init__(self, population_size=100, generations=50, memory=1) -> None:
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.best_moves = {}
        self.memory = memory
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = tuple(map(tuple, game.get_board()))
        
        population = self.__generate_random_valid_move(game)
        
        for _ in range(self.generations):
            population = self.__evolve_population(population, game)
            
        sorted_population = sorted(population, key=lambda move: self.__fitness(move[0], move[1], game), reverse=True)
        best_from_pos, best_slide = sorted_population[0]
        if self.memory:
            if current_board not in self.best_moves:
                self.best_moves[current_board] = (best_from_pos, best_slide)
            else:
                if self.__fitness(best_from_pos, best_slide, game) > self.__fitness(self.best_moves[current_board][0], self.best_moves[current_board][1], game):
                    self.best_moves[current_board] = (best_from_pos, best_slide)
                else:
                    best_from_pos, best_slide = self.best_moves[current_board]

        return best_from_pos, best_slide
    
    def __generate_random_valid_move(self, game: 'Game') -> list(tuple[tuple[int, int], Move]):
        population = []
        for _ in range(self.population_size):
            while True:
                from_pos = (random.randint(0, 4), random.randint(0, 4))
                slide = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
                if self.__verifie_move(from_pos, slide, game):
                    population.append((from_pos, slide))
                    break
        return population
    
    def __evolve_population(self, population: list[tuple[tuple[int, int], Move]], game: 'Game') -> list[tuple[tuple[int, int], Move]]:        
        sorted_population = sorted(population, key=lambda move: self.__fitness(move[0], move[1], game), reverse=True)
        elite_size = int(self.population_size * 0.2)
        elite = sorted_population[:elite_size]
        
        offspring = []
        for _ in range(self.population_size - elite_size):
            parent_1, parent_2 = random.sample(elite, 2)
            crossover_point = random.randint(1, len(parent_1) - 1)
            child1 = parent_1[:crossover_point] + parent_2[crossover_point:]
            offspring.append(child1)
        
        mutated_offspring = [self.__mutate(move) for move in offspring]
        return elite + mutated_offspring
    
    def __mutate(self, move: tuple[tuple[int, int], Move]) -> tuple[tuple[int, int], Move]:
        mutation_rate = 0.1
        if random.uniform(0, 1) < mutation_rate:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            return (from_pos, move)
        return move
    
    def __verifie_move(self, from_pos: tuple[int, int], slide: Move, game: Game) -> bool:
        from_pos = (from_pos[1], from_pos[0])
        player_id = game.get_current_player()
        
        acceptable_take: bool = (
            (from_pos[0] == 0 and from_pos[1] < 5)
            or (from_pos[0] == 4 and from_pos[1] < 5)
            or (from_pos[1] == 0 and from_pos[0] < 5)
            or (from_pos[1] == 4 and from_pos[0] < 5)
        ) and (game.get_board()[from_pos] < 0 or game.get_board()[from_pos] == player_id)
        
        if not acceptable_take:
            return False
        
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
                slide == Move.BOTTOM or slide == Move.RIGHT)
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        return acceptable
    
    def __simulate_move(self, from_pos: tuple[int, int], slide: Move, game: 'Game') -> np.array:
        acceptable = self.__verifie_move(from_pos, slide, game)
        
        # should not happen check move before entering here
        if not acceptable:
            return None
        
        from_pos = (from_pos[1], from_pos[0])
        
        copy_game = deepcopy(game)
        copy_board = copy_game.get_board()
        
        piece = copy_game.get_current_player()
        copy_board[from_pos] = piece
        
        if slide == Move.LEFT:
            for i in range(from_pos[1], 0, -1):
                copy_board[(from_pos[0], i)] = copy_board[(from_pos[0], i - 1)]
            copy_board[(from_pos[0], 0)] = piece
        elif slide == Move.RIGHT:
            for i in range(from_pos[1], copy_board.shape[1] - 1, 1):
                copy_board[(from_pos[0], i)] = copy_board[(from_pos[0], i + 1)]
            copy_board[(from_pos[0], copy_board.shape[1] - 1)] = piece
        elif slide == Move.TOP:
            for i in range(from_pos[0], 0, -1):
                copy_board[(i, from_pos[1])] = copy_board[(i - 1, from_pos[1])]
            copy_board[(0, from_pos[1])] = piece
        elif slide == Move.BOTTOM:
            for i in range(from_pos[0], copy_board.shape[0] - 1, 1):
                copy_board[(i, from_pos[1])] = copy_board[(i + 1, from_pos[1])]
            copy_board[(copy_board.shape[0] - 1, from_pos[1])] = piece
            
        return copy_board
    
    def __fitness(self, from_pos: tuple[int, int], slide: Move, game: 'Game') -> int:
        copy_board = self.__simulate_move(from_pos, slide, game)
        if copy_board is None:
            return -1

        piece = game.get_current_player()
        opponent_piece = 0 if piece == 1 else 1

        def calculate_score(line: np.ndarray) -> int:
            # Calculate score for a row, column, or diagonal
            score = 0
            length = len(line)
            contiguous_count = 0

            for i in range(length):
                if line[i] == piece:
                    contiguous_count += 1
                    score += contiguous_count
                else:
                    contiguous_count = 0
                    if line[i] == opponent_piece:
                        # Penalty for opponent's piece in the line
                        score -= 1

            return score

        row_score = np.sum(np.apply_along_axis(calculate_score, 1, copy_board))
        col_score = np.sum(np.apply_along_axis(calculate_score, 0, copy_board))
        principal_diag_score = calculate_score(np.diag(copy_board))
        secondary_diag_score = calculate_score(np.diag(np.fliplr(copy_board)))

        # Summing up scores with penalties
        score = row_score + col_score + principal_diag_score + secondary_diag_score

        return score
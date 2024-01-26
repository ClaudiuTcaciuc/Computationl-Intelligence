from copy import deepcopy
import random

from game import Player, Move, Game

class GeneticPlayer(Player):
    def __init__(self, population_size=100, generations=50, current_player = 0) -> None:
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.best_moves = {}  # Dictionary to store the best moves for each board
        self.initial_population = self.generate_initial_population()
        self.current_player = current_player

    def generate_initial_population(self) -> list[tuple[tuple[int, int], Move]]:
        population = []
        for _ in range(self.population_size):
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            population.append((from_pos, move))
        return population
    
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
    
    def fitness(self, move: tuple[tuple[int, int], Move], game: 'Game') -> float:
        # TODO: Implement fitness function
        game_copy = deepcopy(game)
        
        from_pos, slide = move
        valid = self.is_valid_move(game_copy, (from_pos[1], from_pos[0]), slide, game.get_current_player())
        #valid = game_copy._Game__move(from_pos, slide, game_copy.get_current_player())
        if valid:
            return self.distance_from_goal(game_copy)
        else:
            return -100
    
    def distance_from_goal(self, game: 'Game') -> float:
        board = game.get_board()
        
        row_score = sum(1 for row in board if all(cell == game.get_current_player() for cell in row))
        col_score = sum(1 for col in zip(*board) if all(cell == game.get_current_player() for cell in col))
        principal_diag_score = sum(1 for i in range(len(board)) if board[i][i] == game.get_current_player())
        secondary_diag_score = sum(1 for i in range(len(board)) if board[i][len(board) - 1 - i] == game.get_current_player())
        
        return row_score + col_score + principal_diag_score + secondary_diag_score
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = tuple(map(tuple, game.get_board()))
        if current_board in self.best_moves:
            return self.best_moves[current_board]
        else:
            population = self.initial_population
            for _ in range(self.generations):
                population = self.evolve_population(population, game)
            sorted_population = sorted(population, key=lambda move: self.fitness(move, game), reverse=True)
            best_move = sorted_population[0]
            self.best_moves[current_board] = best_move
            print("here")
            print("board: ", current_board)
            print("best move: ", best_move)
            print("player: ", game.get_current_player())
            return best_move
    
    def is_valid_move(self, game: 'Game', from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        board = game.get_board()
        
        acceptable_start_pos: bool = (
            (from_pos[0] == 0 and from_pos[1] < 5)
            or (from_pos[0] == 4 and from_pos[1] < 5)
            or (from_pos[1] == 0 and from_pos[0] < 5)
            or (from_pos[1] == 4 and from_pos[0] < 5)
        ) and (board[from_pos] < 0 or board[from_pos] == player_id)

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

        acceptable = (acceptable_top or acceptable_bottom or acceptable_left or acceptable_right) and acceptable_start_pos

        return acceptable

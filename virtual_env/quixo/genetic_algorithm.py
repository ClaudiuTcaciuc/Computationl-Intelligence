from copy import deepcopy
import random
import numpy as np
from game import Player, Move, Game
from utils import verifie_move, simulate_move, fitness


class GeneticPlayer(Player):
    def __init__(self, population_size=100, generations=50, memory=1, test = 0) -> None:
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.best_moves = {}
        self.memory = memory
        self.test = test
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = tuple(map(tuple, game.get_board()))
        
        population = self.__generate_random_valid_move(game)
        
        for _ in range(self.generations):
            population = self.__evolve_population(population, game)
            
        sorted_population = sorted(population, key=lambda move: fitness(move[0], move[1], game), reverse=True)
        best_from_pos, best_slide = sorted_population[0]
        if self.memory:
            if current_board not in self.best_moves:
                self.best_moves[current_board] = (best_from_pos, best_slide)
            else:
                if self.test:
                    best_from_pos, best_slide = self.best_moves[current_board]
                else:
                    if fitness(best_from_pos, best_slide, game) > fitness(self.best_moves[current_board][0], self.best_moves[current_board][1], game):
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
                if verifie_move(from_pos, slide, game):
                    population.append((from_pos, slide))
                    break
        return population
    
    def __evolve_population(self, population: list[tuple[tuple[int, int], Move]], game: 'Game') -> list[tuple[tuple[int, int], Move]]:        
        sorted_population = sorted(population, key=lambda move: fitness(move[0], move[1], game), reverse=True)
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
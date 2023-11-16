import logging
import random
import numpy as np
import pprint as pp
from collections import namedtuple
from copy import deepcopy

Nimply = namedtuple("Nimply", "row num_objects")

# Nim class
class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

# Strategy
def pure_random(state: Nim) -> Nimply:
    """A completely random move"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)

def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    xor = tmp.sum(axis=0) % 2
    return int("".join(str(_) for _ in xor), base=2)

def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked

def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pp.pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
    return ply

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=nim_sum)

def crossover(parent1: Nimply, parent2: Nimply) -> Nimply:
    child_row = parent1.row if random.random() < 0.5 else parent2.row
    child_num_objects = parent2.num_objects if random.random() < 0.5 else parent1.num_objects
    return Nimply(child_row, child_num_objects)

def mutate(move: Nimply, mutation_rate, population):
    if random.random() < mutation_rate:
        mutated_num_objects = random.randint(1, move.num_objects)
        return Nimply(move.row, mutated_num_objects)
    else:
        return crossover(move, random.choice(population))

def evolutionary_strategy(state, generations, population_size, tournament_size, mutation_rate):
    best_move = None
    best_score = float("-inf")
    
    for _ in range(generations):
        population = [pure_random(state) for _ in range(population_size)]
        
        mutated_population = [mutate(move, mutation_rate, population) for move in population]
        selected_population = [tournament_selection(mutated_population, tournament_size) for _ in range(population_size)]
        
        for move in selected_population:
            tmp_state = deepcopy(state)
            tmp_state.nimming(move)
            score = nim_sum(tmp_state)
            
            if score > best_score:
                best_score = score
                best_move = move

    return best_move

def main():
    """" define the game """
    POPULATION_SIZE = 5
    OFFSPRING_SIZE = 10
    TOURNAMENT_SIZE = 2
    MUTATION_RATE = 0.1
    
    logging.getLogger().setLevel(logging.INFO)
    nim = Nim(5)
    logging.info(f"initial state: {nim}")
    
    player = 0

    while nim:
        if player == 0:
            # Use ES for player 0
            ply = evolutionary_strategy(nim, 
                                        generations=OFFSPRING_SIZE, 
                                        population_size=POPULATION_SIZE, 
                                        tournament_size=TOURNAMENT_SIZE, 
                                        mutation_rate=MUTATION_RATE)
        else:
            # Use optimal for player 1
            ply = pure_random(nim)
        
        logging.info(f"player {player} plays {ply}")
        nim.nimming(ply)
        logging.info(f"new state: {nim}")
        
        player = 1 - player
    logging.info(f"player {player} wins!")
        
if __name__ == "__main__":
    main()
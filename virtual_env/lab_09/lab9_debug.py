from random import random, randint, choices

import lab9_lib

class LocalSearchEA:
    def __init__(self, fitness_function, population_size=10, num_generations=100, mutation_rate=0.1, max_stagnation=20):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.max_stagnation = max_stagnation
    
    def initialize_population(self):
        k = 50
        return [choices([0, 1], k=k) for _ in range(self.population_size)]
    
    def mutate(self, individual):
        return [1 - gene if random() < self.mutation_rate else gene for gene in individual]
    
    def crossover(self, parent1, parent2):
        k = randint(1, len(parent1) - 1)
        child1 = parent1[:k] + parent2[k:]
        child2 = parent2[:k] + parent1[k:]
        return child1, child2
    
    def select(self, population):
        return choices(population, k=2, weights=[self.fitness_function(individual) for individual in population])
    
    def run(self):
        population = self.initialize_population()
        best_individual = max(population, key=self.fitness_function)
        best_fitness = self.fitness_function(best_individual)
        stagnation_count = 0

        for generation in range(1, self.num_generations + 1):
            parent1, parent2 = self.select(population)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Replace the worst individuals in the population with children
            population.extend([child1, child2])
            population = sorted(population, key=self.fitness_function, reverse=True)[:self.population_size]

            # Update the best individual and fitness
            current_best_individual = max(population, key=self.fitness_function)
            current_best_fitness = self.fitness_function(current_best_individual)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.max_stagnation:
                print(f"Terminating early due to stagnation after generation {generation}")
                break

            if best_fitness >= 0.99:
                print(f"Terminating early because best fitness reached after generation {generation}")
                break

        print(f"Best individual: {best_individual},\nfitness: {best_fitness},\nfitness calls: {self.fitness_function.calls}")

def main():
    print("Local Search evolutionary algorithm")
    fitness = lab9_lib.make_problem(1)
    ea = LocalSearchEA(fitness, population_size=1, num_generations=1000, mutation_rate=0.1, max_stagnation=100)
    ea.run()

if __name__ == "__main__":
    main()

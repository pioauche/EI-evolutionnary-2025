import numpy as np
import json
import copy
import os
from .Traj3D import Traj3D
from .RotTable import RotTable
from .Individual import Individual

class GeneticOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_fitness = float('inf')
        self.best_solution = None
        self.trajectoire = Traj3D()
        self.pair = 16

    def load_table(self, filename="table.json"):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        table_path = os.path.join(current_dir, filename)
        
        try:
            # Create a RotTable instance as the original table
            self.original_table = Individual(table_path)
            print(f"Successfully loaded table from: {table_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {filename} at {table_path}")
            
    def create_individual(self):
        """Create a mutated version of the original table"""
        new_table = Individual()
        # Appliquer des mutations aléatoires à tous les paramètres
        for dinucleotide in new_table.getTable():
            new_table.addTwist(dinucleotide, np.random.uniform(-30, 30))
            new_table.addWedge(dinucleotide, np.random.uniform(-30, 30))
            new_table.addDirection(dinucleotide, np.random.uniform(-30, 30))
        return new_table
    
    def calculate_fitness(self, ind, sequence):
        """Calculate how circular the structure is"""
        a = 0 # Réduction du poids de l'écart aux angles tabulés
        table = ind.getTable()
        
        self.trajectoire.compute(sequence, ind)
        
        # Get the coordinates
        coords = self.trajectoire.getTraj()
        self.trajectoire.reset()
        
        # Calculate distance between start and end points
        start = np.array(coords[0])
        end = np.array(coords[-1])
        end_to_start = np.linalg.norm(end - start)
        
        # calcul de l'écart au angles tabulés
        norm = 0
        table2 = ind.getTable()
        for cle in table2:
            norm += (table2[cle][0]-table[cle][0])**2 + (table2[cle][1]-table[cle][1])**2 + (table2[cle][2]-table[cle][2])**2
        norm = np.sqrt(norm)/len(table)
        
        # Combine metrics (we want to minimize both)
        return end_to_start + a*norm
    
    def crossover(self, parent1, parent2, type=2):
        """Create a child by combining two parents"""
        child = copy.deepcopy(parent1)
        child.calculate(False)
        table = child.getTable()
        crossover_model = self.__generate_random_tuple(type-1, self.pair)
        dinucleotides = list(table.keys())
        index = 0
        for i,e in enumerate(crossover_model):
            if np.random.random() < 0.5:
                while index < e:
                    dinucleotide = dinucleotides[index]
                    table[dinucleotide] = copy.deepcopy(parent2.getTable()[dinucleotide])
                    index += 1
            index = e
        child.setTable(table)
        
        return self.mutate(child)
    
    def mutate(self, individual:Individual):
        """Apply random mutations to an individual"""
        mutated = False
        while not mutated:  # S'assurer qu'au moins une mutation est appliquée
            for dinucleotide in individual.rot_table:
                if np.random.random() < self.mutation_rate:
                    individual.addTwist(dinucleotide, np.random.uniform(-30, 30))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.addWedge(dinucleotide, np.random.uniform(-30, 30))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.addDirection(dinucleotide, np.random.uniform(-30, 30))
                    mutated = True
        return individual

    def create_new_gen(self, population, type_choosing_parent="tournament", type_matching="random", crossover_type=2):
        """Create a new generation of individuals based on the current population"""
        # Sort population by fitness
        population.sort(key=lambda x: x.getFitness())
        
        # Keep the best 10% of individuals (elitism)
        elite_size = max(1, self.population_size // 10)
        new_population = copy.deepcopy(population[elite_size:])
        
        # Select parents and create offspring until we reach population_size
        while len(new_population) < self.population_size:
            if type_choosing_parent == "tournament":
                # Tournament selection
                tournament_size = 3
                parent1 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                while parent1 == parent2:
                    parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
            else:  # "best" selection
                parent1 = population[np.random.randint(0, len(population) // 2)]
                parent2 = population[np.random.randint(0, len(population) // 2)]
                while parent1 == parent2:
                    parent2 = population[np.random.randint(0, len(population) // 2)]
            
            # Create child through crossover and mutation
            child = self.crossover(parent1, parent2, crossover_type)
            new_population.append(child)
        
        return new_population
    
    def optimize(self, sequence, generations=100):
        """Run the genetic algorithm"""
        # Initialize population with random individuals
        population = []
        for i in range(self.population_size):
            population.append(self.create_individual())  # Tous les individus sont maintenant mutés

        best_fitness_history = []
        generations_without_improvement = 0
        
        for gen in range(generations):
            # Evaluate fitness for each individual
            current_best_fitness = float('inf')
            best_individual = None
            
            for individual in population:
                if not individual.isCalculated():
                    fitness = self.calculate_fitness(individual, sequence)
                    individual.setFitness(fitness)
                    individual.calculate(True)
                
                if individual.getFitness() < current_best_fitness:
                    current_best_fitness = individual.getFitness()
                    best_individual = individual
            
            # Update best solution if we found a better one
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = copy.deepcopy(best_individual)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            best_fitness_history.append(self.best_fitness)
            print(f"Generation {gen}: Best fitness = {self.best_fitness}, Avg fitness = {sum(ind.getFitness() for ind in population)/len(population):.2f}")
            
            # Early stopping if no improvement for many generations
            if generations_without_improvement > 20:
                print("Early stopping: No improvement for 20 generations")
                for popo in population:
                    print(popo.getFitness())
                break
                
            # Create next generation
            population = self.create_new_gen(population)

        return self.best_solution
    
    def save_solution(self, filename='optimized_table.json'):
        """Save the best solution to a file"""
        if self.best_solution:
        # Save the table from the RotTable instance
            with open(filename, 'w') as f:
                json.dump(
                self.best_solution.getTable(), 
                f, 
                indent=4,  # Indentation pour une meilleure lisibilité
                default=lambda o: o.__dict__,
                separators=(',', ': ')  # Ajoute des espaces après les deux-points
            )

    def __generate_random_tuple(self, n, N):
        """Generates a list of indexes to be used for crossover of two parents"""
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)
        random_numbers.sort()
        random_numbers = list(random_numbers)
        return random_numbers+[N] if random_numbers[-1] != N else random_numbers

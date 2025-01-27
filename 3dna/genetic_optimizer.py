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
        # Mutate angles in the table
        table = new_table.getTable()
        for key in table:
            if np.random.random() < self.mutation_rate:
                new_table.AddTwist(key, np.random.normal(-5, 5))  # Add random variation
            if np.random.random() < self.mutation_rate:
                new_table.addWedge(key, np.random.normal(-5, 5))  # Add random variation
            if np.random.random() < self.mutation_rate:
                new_table.addDirection(key, np.random.normal(-5, 5))  # Add random variation
        return new_table
    
    def calculate_fitness(self, ind, sequence):
        """Calculate how circular the structure is"""
        a=1 #poids de l'écart aux angles tabulés
        rot_table = RotTable()
        table=rot_table.getTable()
        traj = Traj3D()
        traj.compute(sequence, table)
        
        # Get the coordinates
        coords = traj.getTraj()
        
        # Calculate distance between start and end points
        start = np.array(coords[0])
        end = np.array(coords[-1])
        end_to_start = np.linalg.norm(end - start)
        
        # calcul de l'écart au angles tabulés
        norm=0
        for clé in ind:
            norm+=(ind[clé][0]-table[clé][0])**2+(ind[clé][1]-table[clé][1])**2+(ind[clé][2]-table[clé][2])**2
        norm=np.sqrt(norm)/len(table)
        
        # Combine metrics (we want to minimize both)
        return end_to_start + a*norm
    
    def crossover(self, parent1, parent2, type=2):
        """Create a child by combining two parents"""
        child = copy.deepcopy(parent1)
        crossover_model = self.__generate_random_tuple(type-1, child.pair)
        dinucleotides = list(child.getTable().keys())
        index = 0
        for i,e in enumerate(crossover_model):
            if np.random.random() < 0.5:
                while index < e:
                    dinucleotide = dinucleotides[index]
                    child[dinucleotide] = copy.deepcopy(parent2[dinucleotide])
                    index += 1
            index = e
        return child
    
    def mutate(self, individual:RotTable):
        """Apply random mutations to an individual"""
        for dinucleotide in individual.rot_table:
            if np.random.random() < self.mutation_rate:
                current_twist = individual.getTwist()
                individual.setTwist(dinucleotide, current_twist + np.random.normal(0, 5))
            if np.random.random() < self.mutation_rate:
                current_wedge = individual.getWedge()
                individual.setWedge(dinucleotide, current_wedge + np.random.normal(0, 5))
            if np.random.random() < self.mutation_rate:
                current_direction = individual.getDirection()
                individual.setDirection(dinucleotide, current_direction + np.random.normal(0, 5))
        return individual
    def create_new_gen(self,population,type_choosing_parent="best",type_matching="random"):
        parents = []
        if type_choosing_parent == "best":
            parents= copy.deepcopy(population)
            parents.sort(key=lambda x: x.getFitness())
            return parents[:self.population_size//2]
        if type_matching == "random":
            for i in range(self.population_size//2):
                parent1 = population[np.random.randint(0,len(population))]
                parent2 = population[np.random.randint(0,len(population))]
                while parent1 == parent2:
                    parent1 = population[np.random.randint(0,len(population))]
                    parent2 = population[np.random.randint(0,len(population))]
                parents.append(self.crossover(parent1, parent2))
            while len(parents) != self.population_size:
                parents.pop()
            return parents
    def optimize(self, sequence, generations=100):
        """Run the genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for gen in range(generations):
            # Evaluate fitness for each individual
            best_idx = float('inf')
            for i,individual in enumerate(population):
                if not individual.isCalculated():
                    fitness = self.calculate_fitness(individual, sequence)
                    individual.setFitness(fitness)
                if individual.getFitness() < best_idx:
                    best_idx = individual.getFitness()
                    id = i
            self.best_fitness=best_idx
            # Keep track of best solution
            self.best_solution = copy.deepcopy(population[id])
            print(f"Generation {gen}: New best fitness = {self.best_fitness}")            
            # Create next generation
            population = self.create_new_gen(population)

        
        return self.best_solution
    
    def save_solution(self, filename='optimized_table.json'):
        """Save the best solution to a file"""
        if self.best_solution:
            # Save the table from the RotTable instance
            with open(filename, 'w') as f:
                json.dump(self.best_solution, f, indent=4)
    
    def __generate_random_tuple(self, n, N):
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)
        random_numbers.sort()
        random_numbers = list(random_numbers)
        return random_numbers+[N] if random_numbers[-1] != N else random_numbers

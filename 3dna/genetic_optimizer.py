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
                new_table.addTwist(key, np.random.normal(-5, 5))  # Add random variation
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
        
        self.trajectoire.compute(sequence, rot_table)
        
        # Get the coordinates
        coords = self.trajectoire.getTraj()
        self.trajectoire.reset()
        # Calculate distance between start and end points
        start = np.array(coords[0])
        end = np.array(coords[-1])
        end_to_start = np.linalg.norm(end - start)
        
        # calcul de l'écart au angles tabulés
        norm=0
        for cle in ind:
            norm+=(ind[cle][0]-table[cle][0])**2+(ind[cle][1]-table[cle][1])**2+(ind[cle][2]-table[cle][2])**2
        norm=np.sqrt(norm)/len(table)
        
        # Combine metrics (we want to minimize both)
        return end_to_start + a*norm
    
    def crossover(self, parent1, parent2):
        """Create a child by combining two parents"""
        child = Individual()
        child.table = {}
        
        for key in parent1:
            if np.random.random() < 0.5:
                child.table[key] = copy.deepcopy(parent1[key])
            else:
                child.table[key] = copy.deepcopy(parent2[key])
        return child
    
    def mutate(self, individual):
        """Apply random mutations to an individual"""
        for key in individual.table:
            for i in [0, 1, 2]:  # Only mutate angle values
                if np.random.random() < self.mutation_rate:
                    individual.table[key][i] += np.random.normal(0, 5)
        return individual
    def create_new_gen(self,population,type_choosing_parent="best",type_matching="random"):
        parents = []
        if type_choosing_parent == "best":
            parents= copy.deepcopy(population)
            parents.sort(key=lambda x: x.getFitness())
            parents = parents[:self.population_size//2]
        if type_matching == "random":
            for i in range(self.population_size//2):
                parent1 = parents[np.random.randint(0,len(parents))]
                parent2 = parents[np.random.randint(0,len(parents))]
                while parent1 == parent2:
                    parent1 = parents[np.random.randint(0,len(parents))]
                    parent2 = parents[np.random.randint(0,len(parents))]
                parents.append(self.crossover(parent1.getTable(), parent2.getTable()))
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
                    fitness = self.calculate_fitness(individual.getTable(), sequence)
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
                json.dump(
                self.best_solution.getTable(), 
                f, 
                indent=4,  # Indentation pour une meilleure lisibilité
                default=lambda o: o.__dict__,
                separators=(',', ': ')  # Ajoute des espaces après les deux-points
            )


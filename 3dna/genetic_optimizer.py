import numpy as np
import json
import copy
from .Traj3D import Traj3D
from .RotTable import RotTable

class GeneticOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_fitness = float('inf')
        self.best_solution = None
        
    def load_table(self, filename='table.json'):
        with open(filename, 'r') as f:
            self.original_table = json.load(f)
            
    def create_individual(self):
        """Create a mutated version of the original table"""
        new_table = copy.deepcopy(self.original_table)
        for key in new_table:
            # Mutate angles (indices 0, 1, 2) with small variations
            for i in [0, 1, 2]:
                if np.random.random() < self.mutation_rate:
                    new_table[key][i] += np.random.normal(0, 10)  # Add random variation
        return new_table
    
    def calculate_fitness(self, table, sequence):
        """Calculate how circular the structure is"""
        rot_table = RotTable()
        rot_table.table = table
        traj = Traj3D()
        traj.compute(sequence, rot_table)
        
        # Get the coordinates
        coords = traj.getTraj()
        
        # Calculate distance between start and end points
        start = np.array(coords[0])
        end = np.array(coords[-1])
        end_to_start = np.linalg.norm(end - start)
        
        # Calculate average radius from center of mass
        center = np.mean(coords, axis=0)
        radii = [np.linalg.norm(point - center) for point in coords]
        radius_variance = np.var(radii)
        
        # Combine metrics (we want to minimize both)
        return end_to_start + radius_variance
    
    def crossover(self, parent1, parent2):
        """Create a child by combining two parents"""
        child = copy.deepcopy(parent1)
        for key in child:
            if np.random.random() < 0.5:
                child[key] = copy.deepcopy(parent2[key])
        return child
    
    def mutate(self, individual):
        """Apply random mutations to an individual"""
        for key in individual:
            for i in [0, 1, 2]:  # Only mutate angle values
                if np.random.random() < self.mutation_rate:
                    individual[key][i] += np.random.normal(0, 5)
        return individual
    
    def optimize(self, sequence, generations=100):
        """Run the genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for gen in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = [self.calculate_fitness(ind, sequence) for ind in population]
            
            # Keep track of best solution
            best_idx = np.argmin(fitness_scores)
            if fitness_scores[best_idx] < self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = copy.deepcopy(population[best_idx])
                print(f"Generation {gen}: New best fitness = {self.best_fitness}")
            
            # Select parents for next generation
            parents = []
            for _ in range(self.population_size):
                tournament = np.random.choice(len(population), 3)
                winner = tournament[np.argmin([fitness_scores[i] for i in tournament])]
                parents.append(population[winner])
            
            # Create next generation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
                
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population
        
        return self.best_solution
    
    def save_solution(self, filename='optimized_table.json'):
        """Save the best solution to a file"""
        if self.best_solution:
            with open(filename, 'w') as f:
                json.dump(self.best_solution, f, indent=4)

import numpy as np
import json
import copy
import os
from abc import ABC, abstractmethod
from .Traj3D import Traj3D
from .RotTable import RotTable
from .Individual import Individual

class Optimizer(ABC):
    def __init__(self, generations=100, population_size=50, mutation_rate=0.1):
        self.generations = generations              # Number of generations
        self.population_size = population_size      # Number of individuals in the population
        self.mutation_rate = mutation_rate          # Probability of mutation per parameter
        self.trajectoire = Traj3D()                 # Instance of Traj3D for trajectory calculations
        self.pair = 16                              # Number of dinucleotide pairs for crossover
        self.best_solution = None                   # Store the best solution found
        self.best_fitness = float('inf')            # Keep track of the best fitness score

    def load_table(self, filename="table.json"):
        # Load the initial table from a JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
        table_path = os.path.join(current_dir, filename)  # Full path to the table file
        
        try:
            # Create a RotTable instance from the loaded file
            self.ind_ref = Individual(table_path)
            print(f"Successfully loaded table from: {table_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {filename} at {table_path}")
        
    def calculate_fitness(self, ind, sequence: str):
        """Calculate the fitness function with normalizations and penalties."""
        a = 0.2  # Initial weight for angle deviation
        rot_table = RotTable()
        table = rot_table.getTable()
        ind_table = ind.getTable()

        self.trajectoire.compute(sequence, ind)
        start, end = self.trajectoire.getTraj()[0], self.trajectoire.getTraj()[-1]
        end_to_start = np.linalg.norm(end - start)
        distance = end_to_start
        # Compute deviation from tabulated angles
        norm = sum(
            (ind_table[key][0] - table[key][0])**2 +
            (ind_table[key][1] - table[key][1])**2 +
            (ind_table[key][2] - table[key][2])**2
            for key in ind_table
        )
        norm = np.sqrt(norm) / len(table)

        # Dynamic normalization to balance values
        max_distance = 10  # Estimated max distance
        max_norm = 5        # Estimated max angle deviation
        end_to_start /= max_distance
        norm /= max_norm

        # Adaptive weight adjustment
        a = 0.2 + (end_to_start ** 2) * 0.5

        # Rotation penalty to prevent unrealistic solutions
        rotation_penalty = sum(abs(ind_table[key][0]) + abs(ind_table[key][1]) + abs(ind_table[key][2]) for key in ind_table)
        rotation_penalty /= (3 * len(table))  # Normalization

        return (end_to_start + a * norm + 0.05 * rotation_penalty,distance)

    def save_solution(self, filename='optimized_table.json'):
        """Save the best solution to a file."""
        if self.best_solution:
            # Save the table from the RotTable instance
            with open(filename, 'w') as f:
                json.dump(
                    self.best_solution.getTable(), 
                    f, 
                    indent=4,  # Pretty-print the JSON
                    default=lambda o: o.__dict__,
                    separators=(',', ': ')  # Add spaces after separators for readability
                )

    @abstractmethod
    def optimize(self, dna_sequence: str, generations=100):
        pass

    @abstractmethod
    def mutate(self, ind: Individual):
        pass

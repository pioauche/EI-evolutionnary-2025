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
        self.original_table = None                  # Original rotation table
        self.best_solution = None  # Store the best solution found
        self.best_fitness = float('inf')  # Keep track of the best fitness score

    def load_table(self, filename="table.json"):
        # Load the initial table from a JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
        table_path = os.path.join(current_dir, filename)  # Full path to the table file
        
        try:
            # Create a RotTable instance from the loaded file
            self.original_table = Individual(table_path)
            print(f"Successfully loaded table from: {table_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {filename} at {table_path}")
        
    def calculate_fitness(self, ind: Individual, dna_sequence: str):
        """Calculate the fitness of an individual based on structural circularity."""
        a = 0.2  # Weight factor for deviation from tabulated angles
        rot_table = RotTable()  # Load the reference rotation table
        table = rot_table.getTable()  # Get the reference table data
        ind_table = ind.getTable()  # Get the individual's table data
        
        self.trajectoire.compute(dna_sequence, ind)  # Compute the 3D trajectory based on the individual
        
        # Get the computed coordinates
        coords = self.trajectoire.getTraj()
        self.trajectoire.reset()  # Reset the trajectory for future calculations
        
        # Calculate the Euclidean distance between the start and end points of the trajectory
        start = np.array(coords[0])
        end = np.array(coords[-1])
        end_to_start = np.linalg.norm(end - start)
        
        # Calculate the deviation from tabulated angles
        norm = 0
        for key in ind_table:
            norm += (ind_table[key][0] - table[key][0])**2 + \
                    (ind_table[key][1] - table[key][1])**2 + \
                    (ind_table[key][2] - table[key][2])**2
        norm = np.sqrt(norm) / len(table)
        
        # Return a combined fitness value (lower is better)
        return end_to_start + a * norm

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

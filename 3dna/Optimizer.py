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
        self.__generations = generations              # Number of generations
        self.__population_size = population_size      # Number of individuals in the population
        self.__mutation_rate = mutation_rate          # Probability of mutation per parameter
        self.__trajectoire = Traj3D()                 # Instance of Traj3D for trajectory calculations
        self.__pair = 16                              # Number of dinucleotide pairs for crossover
        self.__best_solution = None                   # Store the best solution found
        self.__best_fitness = float('inf')            # Keep track of the best fitness score
    ###################
    # WRITING METHODS #
    ###################
    def setMutationRate(self, mutation_rate: float):
        self.__mutation_rate = mutation_rate
    def setBestSolution(self, best_solution: Individual):
        self.__best_solution = best_solution
    def setBestFitness(self, best_fitness: float):
        self.__best_fitness = best_fitness

   ###################
    # READING METHODS #
    ###################
    def getGenerations(self) -> int:
        return self.__generations
    def getPopulationSize(self) -> int:
        return self.__population_size
    def getMutationRate(self) -> float:
        return self.__mutation_rate
    def getBestSolution(self) -> Individual:
        return self.__best_solution
    def getBestFitness(self) -> float:
        return self.__best_fitness
    def getPair(self) -> int:
        return self.__pair
    def getTrajectoire(self) -> Traj3D:
        return self.__trajectoire
    def getIndRef(self) -> Individual:
        return self.__ind_ref
    def load_table(self, filename="table.json"):
        # Load the initial table from a JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
        table_path = os.path.join(current_dir, filename)  # Full path to the table file
        
        try:
            # Create a RotTable instance from the loaded file
            self.__ind_ref = Individual(table_path)
            print(f"Successfully loaded table from: {table_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {filename} at {table_path}")
        
    def calculate_fitness(self, ind, sequence: str):
        """Calculate the fitness function with normalizations and penalties."""
        a = 0.2 # Initial weight for angle deviation
        rot_table = RotTable()
        table = rot_table.getTable()
        ind_table = ind.getTable()

        self.__trajectoire.compute(sequence, ind)
        start, end = self.__trajectoire.getTraj()[0], self.__trajectoire.getTraj()[-1]
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

        """# Dynamic normalization to balance values
        max_distance = 10  # Estimated max distance
        max_norm = 5        # Estimated max angle deviation
        end_to_start /= max_distance
        norm /= max_norm

        # Adaptive weight adjustment
        a = 0.2 + (end_to_start ** 2) * 0.5

        # Rotation penalty to prevent unrealistic solutions
        rotation_penalty = sum(abs(ind_table[key][0]) + abs(ind_table[key][1]) + abs(ind_table[key][2]) for key in ind_table)
        rotation_penalty /= (3 * len(table))  # Normalization
"""
        return (end_to_start + a * norm,distance)

    def save_solution(self, filename='optimized_table.json'):
        """Save the best solution to a file."""
        if self.__best_solution:
            # Save the table from the RotTable instance
            with open(filename, 'w') as f:
                json.dump(
                    self.__best_solution.getTable(), 
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

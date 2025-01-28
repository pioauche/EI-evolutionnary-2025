import numpy as np
import json
import random
import copy
import os
from .Traj3D import Traj3D
from .RotTable import RotTable
from .Individual import Individual

class GeneticOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1):
        # Initialize the genetic optimizer with population size and mutation rate
        self.population_size = population_size  # Number of individuals in the population
        self.mutation_rate = mutation_rate  # Probability of mutation per parameter
        self.best_fitness = float('inf')  # Keep track of the best fitness score
        self.best_solution = None  # Store the best solution found
        self.trajectoire = Traj3D()  # Instance of Traj3D for trajectory calculations
        self.pair = 16  # Number of dinucleotide pairs for crossover
        self.ind_ref = Individual()
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

    def create_individual(self):
        """Create a new individual with random mutations."""
        individu = Individual()
        table = individu.getTable()  # Get the individual's rotation table
        
        # Apply random mutations to all parameters of the individual's table
        for dinucleotide in table:
            dinu = table[dinucleotide]
            individu.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))
            individu.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
            individu.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
        return individu

    def calculate_fitness(self, ind: Individual, sequence: str):
        """Calculate the fitness of an individual based on structural circularity."""
        a = 0.2  # Weight factor for deviation from tabulated angles
        rot_table = RotTable()  # Load the reference rotation table
        table = rot_table.getTable()  # Get the reference table data
        ind_table = ind.getTable()  # Get the individual's table data
        
        self.trajectoire.compute(sequence, ind)  # Compute the 3D trajectory based on the individual
        
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

    def crossover(self, parent1: Individual, parent2: Individual, crossover_type=2,test=False):
        """Create a child individual by combining two parents."""
        if test==True:
            child = copy.deepcopy(parent1)
        else:
            child = copy.deepcopy(parent1)  # Start with a copy of the first parent
            child.setCalculated(False)  # Mark as not calculated
            table = child.getTable()  # Access the child's rotation table
            crossover_model = self.__generate_random_tuple(crossover_type-1, self.pair)  # Generate crossover points
            dinucleotides = list(table.keys())  # List of dinucleotide keys in the table
            
            index = 0
            for i, e in enumerate(crossover_model):
                if np.random.random() < 0.5:
                    # Swap segments between parents based on crossover model
                    while index < e:
                        dinucleotide = dinucleotides[index]
                        table[dinucleotide] = copy.deepcopy(parent2.getTable()[dinucleotide])
                        index += 1
                index = e  # Update index for the next segment
            child.setTable(table)  # Update the child's table
            
        return child  # Apply mutation to the child

    def mutate(self, individual: Individual):
        """Apply random mutations to an individual."""
        table = self.ind_ref.getTable()  # Get the individual's rotation table
        # Ensure at least one mutation occurs
        for dinucleotide in table:
            dinu = table[dinucleotide]
            if np.random.random() < self.mutation_rate:
                individual.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))
            if np.random.random() < self.mutation_rate:
                individual.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
            if np.random.random() < self.mutation_rate:
                individual.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
        
        return individual  # Return the mutated individual

    def create_new_gen(self, population, type_choosing_parent="best", type_matching="random", crossover_type=2):
        """Create a new generation of individuals based on the current population"""
        b=0.25 #proportion d'individus selectionnés
        if type_choosing_parent == "tournoi":
            parents=[]
            populationbis = copy.deepcopy(population)
            for i in range(int(b*self.population_size)):
                tournament_size = 3
                parent = min(np.random.choice(populationbis, tournament_size), key=lambda x: x.getFitness())
                populationbis.remove(parent)
                parents.append(parent)
        elif type_choosing_parent == "selection par rang":
            # Rank-based selection of parents
            parents = []
            population.sort(key=lambda x: x.getFitness())
            poids=[self.population_size-k for k in range(self.population_size)]
            poids_temp = poids[:]  # Copie temporaire des poids
            for _ in range(int(b*self.population_size)):
                total = sum(poids_temp)
                if total == 0:
                    raise ValueError("No weights left to select more parents.")
                seuil = random.uniform(0, total)
                cumul = 0
                for i, poids in enumerate(poids_temp):
                    cumul += poids
                    if seuil <= cumul:
                        parents.append(population[i])
                        poids_temp[i] = 0
                        break

        elif type_choosing_parent == "selection par roulette":
            # Roulette wheel selection
            parents = []
            liste=[x.getFitness() for x in population]
            max=max(liste)
            poids_temp = [max-x for x in liste]  # Copie temporaire des poids
            for _ in range(int(b*self.population_size)):
                total = sum(poids_temp)
                if total == 0:
                    raise ValueError("No weights left to select more parents.")
                seuil = random.uniform(0, total)
                cumul = 0
                for i, poids in enumerate(poids_temp):
                    cumul += poids
                    if seuil <= cumul:
                        parents.append(population[i])
                        poids_temp[i] = 0
                        break

        elif type_choosing_parent == "best":
            # Select the best individuals as parents
            parents = copy.deepcopy(population)
            parents.sort(key=lambda x: x.getFitness())
            parents = parents[:int(b*self.population_size)]
        child=copy.deepcopy(parents)
        if type_matching == "random":
            while len(child)<self.population_size:
                parent1 = parents[np.random.randint(0,len(parents))]
                parent2 = parents[np.random.randint(0,len(parents))]
                while parent1 == parent2:
                    parent2 = parents[np.random.randint(0, len(parents))]
                child.append(self.crossover(parent1, parent2, crossover_type))
        elif type_matching == "tournament":
            # Tournament selection
            while len(child)<self.population_size:
                tournament_size = 3
                parent1 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                while parent1 == parent2:
                    parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                child.append(self.crossover(parent1, parent2, crossover_type))
        elif type_matching == "meritocratie":
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    child1 = self.crossover(parents[i], parents[j], crossover_type)
                    self.mutate(child1)
                    child.append(child1)
                    if len(child) >= self.population_size:
                        break
        if len(child) > self.population_size:
            child.pop()
        for ind in child[1:]:#on permet a tous le mondde de muter sauf le premier

            ind = self.mutate(ind)
        return child  # Return the new generation

    def optimize(self, sequence: str, generations=100):
        """Run the genetic algorithm."""
        # Initialize population with random individuals
        population = []
        population.append(Individual())
        for i in range(1,self.population_size):
            population.append(self.create_individual())

        best_fitness_history = []  # Track the best fitness over generations
        generations_without_improvement = 0  # Track stagnation in progress

        for gen in range(generations):
            # Evaluate fitness for each individual
            current_best_fitness = float('inf')
            best_individual = None

            for individual in population:
                if not individual.isCalculated():
                    fitness = self.calculate_fitness(individual, sequence)
                    individual.setFitness(fitness)
                    individual.setCalculated(True)

                if individual.getFitness() < current_best_fitness:
                    current_best_fitness = individual.getFitness()
                    best_individual = individual

            # Update best solution if a better one is found
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = copy.deepcopy(best_individual)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            #mutation adaptative selon le nombre d'iterations sans amélioration
            if generations_without_improvement > 5:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.05)  # Augmente la mutation
            else:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)  # Réduit légèrement
            best_fitness_history.append(self.best_fitness)
            print(f"Generation {gen}: Best fitness = {self.best_fitness}, Avg fitness = {sum(ind.getFitness() for ind in population)/len(population):.2f}")

            # Early stopping if no improvement for many generations
            if generations_without_improvement > 40:
                print("Early stopping: No improvement for 20 generations")
                break

            # Create the next generation
            population = self.create_new_gen(population,type_choosing_parent="best",type_matching="random",crossover_type=2)

        return self.best_solution  # Return the best solution found

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

    def __generate_random_tuple(self, n, N):
        """Generate a list of indexes for crossover of two parents."""
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)  # Select unique random indices
        random_numbers.sort()  # Sort indices in ascending order
        random_numbers = list(random_numbers)
        return random_numbers + [N] if random_numbers[-1] != N else random_numbers
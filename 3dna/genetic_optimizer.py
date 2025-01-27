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
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_fitness = float('inf')
        self.best_solution = None
        self.trajectoire = Traj3D()
        self.pair= 16
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
        self.mutate(new_table)
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
    
    def crossover(self, parent1, parent2, crossover_type=2):
        """Create a child by combining two parents"""
        child = copy.deepcopy(parent1)
        child.calculate(False)
        table = child.getTable()
        crossover_model = self.__generate_random_tuple(crossover_type-1, self.pair)
        dinucleotides = list(table.keys())
        index = 0
        for i,e in enumerate(crossover_model):
            if np.random.random() < 0.5:
                while index < e:
                    dinucleotide = dinucleotides[index]
                    table[dinucleotide] = copy.deepcopy(parent2.getTable()[dinucleotide])
                    index += 1
            index = e
        return child
    
    def mutate(self, individual:RotTable):
        """Apply random mutations to an individual"""
        for dinucleotide in individual.rot_table:
            if np.random.random() < self.mutation_rate:
                individual.setTwist(dinucleotide, np.random.normal(0, 5))
            if np.random.random() < self.mutation_rate:
                individual.setWedge(dinucleotide, np.random.normal(0, 5))
            if np.random.random() < self.mutation_rate:
                individual.setDirection(dinucleotide, np.random.normal(0, 5))
        return individual

    def create_new_gen(self, population, type_choosing_parent="best", type_matching="random", crossover_type=2):
        """Create a new generation of individuals based on the current population"""
        parents = []
        if type_choosing_parent == "selestion par roulette":
            parents = []
            liste=[x.getFitness() for x in population]


            poids_temp = liste[:]  # Copie temporaire des poids
            for _ in range(self.population_size//2):
                total = sum(poids_temp)
                if total == 0:
                    raise ValueError("Impossible de tirer plus d'indices : les poids sont épuisés.")
                
                seuil = random.uniform(0, total)
                cumul = 0
                for i, poids in enumerate(poids_temp):
                    cumul += poids
                    if seuil <= cumul:
                        parents.append(population[i])  # Ajouter l'indice tiré
                        poids_temp[i] = 0  # Réduire le poids à zéro (sans retour)
                        break
        if type_choosing_parent == "best":
            parents = copy.deepcopy(population)
            parents.sort(key=lambda x: x.getFitness())
            parents = parents[:self.population_size//2]
        child=copy.deepcopy(parents)
        if type_matching == "random":
            for i in range(self.population_size//2):
                parent1 = parents[np.random.randint(0,len(parents))]
                parent2 = parents[np.random.randint(0,len(parents))]
                while parent1 == parent2:
                    parent1 = parents[np.random.randint(0,len(parents))]
                    parent2 = parents[np.random.randint(0,len(parents))]
                child.append(self.crossover(parent1.getTable(), parent2.getTable(), crossover_type))
            while len(child) > self.population_size:
                child.pop()
        return child
    
    def optimize(self, sequence, generations=100):
        """Run the genetic algorithm"""
        # Initialize population
        population = []
        population.append(Individual())
        for i in range(self.population_size-1):
            temp = Individual()
            for dinucleotide in temp.getTable():
                temp.addTwist(dinucleotide,np.random.randint(-20,21))
                temp.addWedge(dinucleotide,np.random.randint(-5,6))
                temp.addDirection(dinucleotide,np.random.randint(-30,31))
            population.append(temp)
        for gen in range(generations):
            # Evaluate fitness for each individual
            best_idx = float('inf')
            for i,individual in enumerate(population):
                if not individual.isCalculated():
                    fitness = self.calculate_fitness(individual.getTable(), sequence)
                    individual.setFitness(fitness)
                    individual.calculate(True)
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

    def __generate_random_tuple(self, n, N):
        """Generates a list of indexes to be used for crossover of two parents"""
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)
        random_numbers.sort()
        random_numbers = list(random_numbers)
        return random_numbers+[N] if random_numbers[-1] != N else random_numbers

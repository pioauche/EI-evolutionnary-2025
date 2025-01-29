import numpy as np
import json
import random
import copy
import os
from .Traj3D import Traj3D
from .RotTable import RotTable
from .Individual import Individual
from .Optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, generations=100, population_size=50, mutation_rate=0.1):
        # Initialize the genetic optimizer with population size and mutation rate
        super().__init__(generations, population_size, mutation_rate)

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

    def crossover(self, parent1: Individual, parent2: Individual, crossover_type=2):
        """Create a child individual by combining two parents."""
        if test==True:
            child = copy.deepcopy(parent1)  # Start with a copy of the first parent
            child.setCalculated(False)  # Mark as not calculated
            table = child.getTable()  # Access the child's rotation table
            parent2_table = parent2.getTable()  # Access parent2's table
            dinucleotides = list(table.keys())  # List of dinucleotide keys in the table

            # Uniform crossover: for each key, randomly choose from parent1 or parent2
            for dinucleotide in dinucleotides:
                alpha = np.random.random()  # Alpha entre 0 et 1
                for i in range(3):
                    table[dinucleotide][i] = alpha * parent1.getTable()[dinucleotide][i] + (1 - alpha) * parent2_table[dinucleotide][i]            
            child.setTable(table)  # Update the child's table
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

    def create_new_gen(self, population,gen, type_choosing_parent="best", type_matching="random", crossover_type=2,):
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
                seuil = np.random.uniform(0, total)
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
            maxi=max(liste)
            poids_temp = [maxi-x for x in liste]  # Copie temporaire des poids
            for _ in range(int(b*self.population_size)):
                total = sum(poids_temp)
                if total == 0:
                    raise ValueError("No weights left to select more parents.")
                seuil = np.random.uniform(0, total)
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
        if gen >10:
            for i in range(len(parents) // 4):  # Remplace 25% de la population
                population[len(parents)-i] = self.create_individual()
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
        for ind in child[int(0.1*self.population_size):]:#on permet a tous le mondde de muter sauf les 10% meilleurs
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
            """if generations_without_improvement > 5:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.05)  # Augmente la mutation
            else:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)  # Réduit légèrement"""
            best_fitness_history.append(self.best_fitness)
            print(f"Generation {gen}: Best fitness = {self.best_fitness}, Avg fitness = {sum(ind.getFitness() for ind in population)/len(population):.2f}")

            # Early stopping if no improvement for many generations
            if generations_without_improvement > 40:
                print("Early stopping: No improvement for 40 generations")
                break

            # Create the next generation
            population = self.create_new_gen(population,type_choosing_parent="best",type_matching="random",crossover_type=2,gen=generations_without_improvement)

        return self.best_solution  # Return the best solution found

    def __generate_random_tuple(self, n, N):
        """Generate a list of indexes for crossover of two parents."""
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)  # Select unique random indices
        random_numbers.sort()  # Sort indices in ascending order
        random_numbers = list(random_numbers)
        return random_numbers + [N] if random_numbers[-1] != N else random_numbers
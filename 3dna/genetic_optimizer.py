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
        individu = Individual()
        table = individu.getTable()
        # Appliquer des mutations aléatoires à tous les paramètres
        for dinucleotide in table:
            individu.addTwist(dinucleotide, np.random.uniform(-table[dinucleotide][3], table[dinucleotide][3]))
            individu.addWedge(dinucleotide, np.random.uniform(-table[dinucleotide][4], table[dinucleotide][4]))
            individu.addDirection(dinucleotide, np.random.uniform(-table[dinucleotide][5], table[dinucleotide][5]))
        return individu
    
    def calculate_fitness(self, ind:Individual, sequence):
        """Calculate how circular the structure is"""
        a = 0.2 # Réduction du poids de l'écart aux angles tabulés
        rot_table = RotTable()
        table = rot_table.getTable()
        ind_table = ind.getTable()
        
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
        for key in ind_table:
            norm += (ind_table[key][0]-table[key][0])**2 + (ind_table[key][1]-table[key][1])**2 + (ind_table[key][2]-table[key][2])**2
        norm = np.sqrt(norm)/len(table)
        
        # Combine metrics (we want to minimize both)
        return end_to_start + a*norm
    
    def crossover(self, parent1, parent2, crossover_type=2):
        """Create a child by combining two parents"""
        child = copy.deepcopy(parent1)
        child.setCalculated(False)
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
        child.setTable(table)
        
        return self.mutate(child)
    
    def mutate(self, individual:Individual):
        """Apply random mutations to an individual"""
        mutated = False
        table = individual.getTable()
        while not mutated:  # S'assurer qu'au moins une mutation est appliquée
            for dinucleotide in table:
                if np.random.random() < self.mutation_rate:
                    individual.addTwist(dinucleotide, np.random.uniform(-table[dinucleotide][3], table[dinucleotide][3]))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.addWedge(dinucleotide, np.random.uniform(-table[dinucleotide][4], table[dinucleotide][4]))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.addDirection(dinucleotide, np.random.uniform(-table[dinucleotide][5], table[dinucleotide][5]))
                    mutated = True
        return individual

    def create_new_gen(self, population, type_choosing_parent="best", type_matching="random", crossover_type=2):
        """Create a new generation of individuals based on the current population"""
        b=0.5 #proportion d'individus selectionnés
        if type_choosing_parent == "tournoi":
            parents=[]
            populationbis = copy.deepcopy(population)
            for i in range(int(b*self.population_size)):
                tournament_size = 3
                parent = min(np.random.choice(populationbis, tournament_size), key=lambda x: x.getFitness())
                populationbis.remove(parent)
                parents.append(parent)
        if type_choosing_parent == "selection par rang":
            parents = []
            population.sort(key=lambda x: x.getFitness())
            poids=[k+1 for k in range(self.population_size)]
            poids_temp = poids[:]  # Copie temporaire des poids
            for _ in range(int(b*self.population_size)):
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
            parents = parents[:int(b*self.population_size)]
        child=copy.deepcopy(parents)
        if type_matching == "random":
            for i in range(int(b*self.population_size)):
                parent1 = parents[np.random.randint(0,len(parents))]
                parent2 = parents[np.random.randint(0,len(parents))]
                while parent1 == parent2:
                    parent2 = parents[np.random.randint(0,len(parents))]
                child.append(self.crossover(parent1, parent2, crossover_type))
        if type_matching == "tournament":
            # Tournament selection
            for i in range(int(b*self.population_size)):
                tournament_size = 3
                parent1 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                while parent1 == parent2:
                    parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                child.append(self.crossover(parent1, parent2, crossover_type))
        return child
    
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
                    individual.setCalculated(True)
                
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
                break
                
            # Create next generation
            population = self.create_new_gen(population)
        for popo in population:
            print(popo.getFitness())
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

import numpy as np
import json
import random
import copy
import os
from Traj3D import Traj3D
from RotTable import RotTable
from Individual import Individual
from Optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, generations=100, population_size=50, mutation_rate=0.1):
        # Initialize the genetic optimizer with population size and mutation rate
        super().__init__(generations, population_size, mutation_rate)
        self.load_table()  # Load the reference table

    def create_individual(self):
        """Create a new individual with random mutations."""
        individu = Individual()
        table = individu.getTable()  # Get the individual's rotation table
        
        # Apply random mutations to all parameters of the individual's table
        for dinucleotide in table:
            dinu = table[dinucleotide]
            individu.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))#we stay in the born given by the reference individual
            individu.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
            individu.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
        return individu

    def crossover(self, parent1: Individual, parent2: Individual, crossover_type=2,test=False):
        """Create a child individual by combining two parents."""
        child = copy.deepcopy(parent1)  # Start with a copy of the first parent
        child.setCalculated(False)  # Mark as not calculated (the children's fitness has never been calculated)
        table = child.getTable()  # Access the child's rotation table

        if test==True:
            parent2_table = parent2.getTable()  # Access parent2's table
            dinucleotides = list(table.keys())  # List of dinucleotide keys in the table

            # Uniform crossover: for each key, randomly choose from parent1 or parent2
            for dinucleotide in dinucleotides:
                alpha = np.random.random()  # Alpha entre 0 et 1
                for i in range(3):
                    #we take a children which is on the 'line' between the two parents (alpha*parent1+(1-alpha)*parent2)
                    table[dinucleotide][i] = alpha * parent1.getTable()[dinucleotide][i] + (1 - alpha) * parent2_table[dinucleotide][i]            
            child.setTable(table)  # Update the child's table
        else:
            crossover_model = self.__generate_random_tuple(crossover_type-1, self.getPair())  # Generate crossover points
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
        table = self.getIndRef().getTable()  # Get the individual's rotation table
        for dinucleotide in table:
            dinu = table[dinucleotide]
            if np.random.random() < self.getMutationRate():#we mutate the twist, wedge and direction of the dinucleotide with a probability of mutation_rate
                individual.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))#we stay in the born given by the reference individual
            if np.random.random() < self.getMutationRate():
                individual.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
            if np.random.random() < self.getMutationRate():
                individual.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
        
        return individual  # Return the mutated individual

    def create_new_gen(self, population,gen, type_choosing_parent="best", type_matching="random", crossover_type=2,):
        """Create a new generation of individuals based on the current population"""
        b=0.25 #proportion of the population that will be parents 
        if type_choosing_parent == "tournoi":#we choose the parents with a tournament
            parents=[]
            populationbis = copy.deepcopy(population)
            for i in range(int(b*self.getPopulationSize())):
                tournament_size = 3#size of the tournament (it can be changed)
                parent = min(np.random.choice(populationbis, tournament_size), key=lambda x: x.getFitness())#we choose randomly two individuals and we keep the best one
                populationbis.remove(parent)#we remove the best (to avoid having the same parent twice)
                parents.append(parent)#we add the best to the parents
        elif type_choosing_parent == "selection par rang":  # Rank-based selection of parents
            parents = []  
            # Step 1: Sort the population based on fitness (ascending order)
            population.sort(key=lambda x: x.getFitness())  # The best individuals (lowest fitness) are at the beginning

            # Step 2: Assign ranks (weights) to individuals based on their position in the sorted list
            poids = [self.getPopulationSize() - k for k in range(self.getPopulationSize())]  # Higher ranks (weights) for better individuals
            poids_temp = poids[:]  # Copy of the weights list to modify during selection

            # Step 3: Select a fraction (b * population size) of individuals as parents
            for _ in range(int(b * self.getPopulationSize())):
                total = sum(poids_temp)  # Compute the total sum of available weights
                
                # Step 3.1: If all weights are zero, selection is no longer possible
                if total == 0:
                    raise ValueError("No weights left to select more parents.")

                # Step 3.2: Generate a random threshold between 0 and the total weight
                seuil = np.random.uniform(0, total)  
                cumul = 0  # Cumulative sum to determine the selected parent
                
                # Step 3.3: Iterate through the weights to find the selected parent
                for i, poids in enumerate(poids_temp):
                    cumul += poids  # Increment the cumulative sum
                    if seuil <= cumul:  # If the threshold is reached, select this parent
                        parents.append(population[i])  # Add the selected parent to the list
                        poids_temp[i] = 0  # Set its weight to zero to prevent reselection
                        break  # Exit the loop once a parent is selected


        elif type_choosing_parent == "selection par roulette":  # Roulette wheel selection
            parents = []  
            
            # Step 1: Extract fitness values of all individuals
            liste = [x.getFitness() for x in population]  
            
            # Step 2: Compute weights for selection (inverse fitness)
            maxi = max(liste)  # Find the worst fitness (highest value)
            poids_temp = [maxi - x for x in liste]  # Higher fitness values get lower weights, ensuring the best individuals are favored

            # Step 3: Select a fraction (b * population size) of individuals as parents
            for _ in range(int(b * self.getPopulationSize())):
                total = sum(poids_temp)  # Compute the total sum of available weights

                # Step 3.1: If all weights are zero, selection is no longer possible
                if total == 0:
                    raise ValueError("No weights left to select more parents.")

                # Step 3.2: Generate a random threshold between 0 and the total weight
                seuil = np.random.uniform(0, total)  
                cumul = 0  # Cumulative sum to determine the selected parent
                
                # Step 3.3: Iterate through the weights to find the selected parent
                for i, poids in enumerate(poids_temp):
                    cumul += poids  # Increment the cumulative sum
                    if seuil <= cumul:  # If the threshold is reached, select this parent
                        parents.append(population[i])  # Add the selected parent to the list
                        poids_temp[i] = 0  # Set its weight to zero to prevent reselection
                        break  # Exit the loop once a parent is selected


        elif type_choosing_parent == "best":
            # Select the  b best individuals as parents
            parents = copy.deepcopy(population)
            parents.sort(key=lambda x: x.getFitness())
            parents = parents[:int(b*self.getPopulationSize())]
        if gen >10  and np.random.random() < 0.15:#if the algorithm is stuck in a local minimum, we add new individuals to the population with a probability of 15%
            for i in range(len(parents) // 4):   #we replace 25% of the population by new individuals to ensure diversity
                population[len(parents)-i] = self.create_individual()
        child=copy.deepcopy(parents)

        if type_matching == "random":
            while len(child)<self.getPopulationSize():
                #we select two parents randomly
                parent1 = parents[np.random.randint(0,len(parents))]
                parent2 = parents[np.random.randint(0,len(parents))]
                while parent1 == parent2:#we ensure that the two parents are different
                    parent2 = parents[np.random.randint(0, len(parents))]
                child.append(self.crossover(parent1, parent2, crossover_type))
        elif type_matching == "tournament":
            # Tournament selection
            while len(child)<self.getPopulationSize():
                tournament_size = 3
                #we select two parents by a tournament-based selection
                parent1 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                while parent1 == parent2:#we ensure that the two parents are different
                    parent2 = min(np.random.choice(population, tournament_size), key=lambda x: x.getFitness())
                child.append(self.crossover(parent1, parent2, crossover_type))#we create a child by crossing the two parents
        elif type_matching == "meritocratie":#we make  the better ones have more children (a test that we made that seems interesting)
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    child1 = self.crossover(parents[i], parents[j], crossover_type)#we create a child by crossing the two parents
                    child.append(child1)
                    if len(child) >= self.getPopulationSize():#we stop if we have the right number of population
                        break
        if len(child) > self.getPopulationSize():#we remove if we have more than the population size (it can happen if we have an odd number of parents)
            child.pop()
        for ind in child[int(0.1*self.getPopulationSize()):]:#we allow 90% worst individuals to mutate
            ind = self.mutate(ind)
        return child  # Return the new generation

    def optimize(self, sequence: str, generations=100):
        """Run the genetic algorithm."""
        # Initialize population with random individuals
        population = []
        population.append(Individual())#we add the reference individual
        for i in range(1,self.getPopulationSize()):#we start with one individual already created
            population.append(self.create_individual())

        generations_without_improvement = 0  # Track stagnation in progress

        for gen in range(generations):
            # Evaluate fitness for each individual
            current_best_fitness = float('inf')
            best_individual = None

            for individual in population:
                if not individual.isCalculated():#only calculate if not already calculated (for optimization)
                    fitness = self.calculate_fitness(individual, sequence)
                    
                    individual.setFitness(fitness[0])
                    individual.setDistance(fitness[1])
                    individual.setCalculated(True)

                if individual.getFitness() < current_best_fitness:
                    current_best_fitness = individual.getFitness()
                    best_individual = individual

            # Update best solution if a better one is found
            if current_best_fitness < self.getBestFitness():
                self.setBestFitness(current_best_fitness) 
                self.__distance = best_individual.getDistance()
                self.setBestSolution(copy.deepcopy(best_individual)) 
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            #mutation adaptative selon le nombre d'iterations sans amélioration
            """if generations_without_improvement > 5:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.05)  # Augmente la mutation
            else:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)  # Réduit légèrement"""
            print(f"Generation {gen}: Best fitness = {self.getBestFitness():.2f}, Distance = {self.__distance:.2f}")

            # Early stopping if no improvement for many generations
            if generations_without_improvement > 40:
                print("Early stopping: No improvement for 40 generations")
                break

            # Create the next generation
            population = self.create_new_gen(population,type_choosing_parent="best",type_matching="random",crossover_type=2,gen=generations_without_improvement)

        return self.getBestSolution()  # Return the best solution found

    def __generate_random_tuple(self, n, N):
        """Generate a list of indexes for crossover of two parents."""
        if n > N:
            raise ValueError(f"n must be less than or equal to {N-1}")
        random_numbers = np.random.choice(np.arange(1, N-1), n, replace=False)  # Select unique random indices
        random_numbers.sort()  # Sort indices in ascending order
        random_numbers = list(random_numbers)
        return random_numbers + [N] if random_numbers[-1] != N else random_numbers
    







    
import numpy as np
import copy
from .Optimizer import Optimizer
from .Individual import Individual

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, generations=100, population_size=50, mutation_rate=0.1, kmax=10000, emax=1, initial_temp=100):
        super().__init__(generations, population_size, mutation_rate)
        self.kmax = kmax
        self.emax = emax
        self.initial_temp = initial_temp

    def mutate(self, individual: Individual):
        """Apply random mutations to an individual."""
        mutated = False  # Track if any mutation occurred
        table = individual.getTable()  # Access the individual's rotation table
        
        while not mutated:
            # Ensure at least one mutation occurs
            for dinucleotide in table:
                dinu = table[dinucleotide]
                if np.random.random() < self.mutation_rate:
                    individual.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
                    mutated = True
                if np.random.random() < self.mutation_rate:
                    individual.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
                    mutated = True
        
        return individual

    def optimize(self, dna_sequence:str, generations=100):
        """
            initial_temp (dict): Table initiale recensant les températures initiales.
            kmax (int): Nombre maximum d'itérations.
            emax (float): Valeur seuil de score acceptable pour arrêter l'algorithme.
            initial_temp (float): Température initiale pour le recuit simulé.
        """
        # Initialisation
        current_individual = copy.deepcopy(self.ind_ref)
        current_fitness = self.calculate_fitness(current_individual, dna_sequence)
        self.best_solution = copy.deepcopy(current_individual)
        self.best_fitness = current_fitness
        print(f"Initial fitness: {current_fitness}")
        k = 0

        while k < self.kmax and current_fitness > self.emax:
            # Générer un voisin en modifiant une valeur de la table
            neighbor_individual = self.mutate(copy.deepcopy(current_individual))

            # Calculer le score du voisin
            neighbor_fitness = self.calculate_fitness(neighbor_individual, dna_sequence)

            # Accepter ou rejeter le voisin
            temperature = self.initial_temp * (1 - k / self.kmax)  # Fonction de refroidissement linéaire
            if neighbor_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - neighbor_fitness) / temperature):
                current_individual = neighbor_individual
                current_fitness = neighbor_fitness

            # Mise à jour du meilleur résultat
            if current_fitness < self.best_fitness:
                self.best_solution = copy.deepcopy(current_individual)
                self.best_fitness = current_fitness

            if k % 10 == 0:
                print(f"Best fitness: {self.best_fitness} for iteration {k}")

            # Passer à l'itération suivante
            k += 1

        print(f"Best fitness: {self.best_fitness} for iteration {self.kmax}")
        return self.best_solution

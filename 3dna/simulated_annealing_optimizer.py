import numpy as np
import copy
from .Optimizer import Optimizer
from .Individual import Individual

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, generations=100, population_size=50, mutation_rate=0.1, individual=Individual(), kmax=10000, emax=2, initial_temp=10):
        super().__init__(generations, population_size, mutation_rate)
        self.individual = individual
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

    def optimize(self, dna_sequence:str, generations=100):
        """
            initial_temp (dict): Table initiale recensant les températures initiales.
            kmax (int): Nombre maximum d'itérations.
            emax (float): Valeur seuil de score acceptable pour arrêter l'algorithme.
            initial_temp (float): Température initiale pour le recuit simulé.
        """
        # Initialisation
        current_individual = copy.deepcopy(self.individual)
        current_fitness = self.calculate_fitness(current_individual, dna_sequence)
        self.best_solution = copy.deepcopy(current_individual)
        self.best_fitness = current_fitness
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

            if k % 100 == 0:
                print(f"Best fitness: {self.best_fitness} for iteration {k}")

            # Passer à l'itération suivante
            k += 1

        print(f"Best fitness: {self.best_fitness}")
        return self.best_solution

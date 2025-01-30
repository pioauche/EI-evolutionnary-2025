import numpy as np
import copy
from .Optimizer import Optimizer
from .Individual import Individual

class SimulatedAnnealingOptimizer(Optimizer):
    """Simulated Annealing optimizer for DNA structural circularity. Subclass of Optimizer."""

    def __init__(self, generations=100, population_size=50, mutation_rate=0.1, kmax=5000, emax=1, initial_temp=100):
        """Initialize the simulated annealing optimizer with population size and mutation rate. Uses the constructor of the parent class."""

        super().__init__(generations, population_size, mutation_rate)
        self.kmax = kmax                    # Maximum number of iterations
        self.emax = emax                    # Acceptable score threshold to stop the algorithm
        self.initial_temp = initial_temp    # Initial temperature for simulated annealing

    def mutate(self, individual: Individual):
        """Apply random mutations to an individual. Returns the mutated individual."""

        mutated = False                 # Track if any mutation occurred
        table = individual.getTable()   # Access the individual's rotation table
        
        while not mutated:              # Ensure at least one mutation occurs
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

    def optimize(self, dna_sequence:str):
        """Run the simulated annealing optimization algorithm. Returns the best solution found."""

        # Initialisation
        current_individual = copy.deepcopy(self.ind_ref)
        current_fitness = self.calculate_fitness(current_individual, dna_sequence)
        self.best_solution = copy.deepcopy(current_individual)
        self.best_fitness = current_fitness
        # print(f"Initial fitness: {current_fitness}")
        k = 0

        while k < self.kmax and current_fitness > self.emax:
            # Generate a neighbor of the current solution by mutation
            neighbor_individual = self.mutate(copy.deepcopy(current_individual))

            # Calculate the fitness of the neighbor
            neighbor_fitness = self.calculate_fitness(neighbor_individual, dna_sequence)

            # Accept the neighbor if it has a better fitness or with a certain probability
            temperature = self.initial_temp * (1 - k / self.kmax)   # Decrease the temperature over time, linearly in this case
            if neighbor_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - neighbor_fitness) / temperature):
                current_individual = neighbor_individual
                current_fitness = neighbor_fitness

            # Keep track of the best solution found
            if current_fitness < self.best_fitness:
                self.best_solution = copy.deepcopy(current_individual)
                self.best_fitness = current_fitness

            # if k % 10 == 0:
            #     print(f"Best fitness: {self.best_fitness} for iteration {k}")

            
            k += 1  # Increment the iteration counter    

        print(f"Best fitness: {self.best_fitness} for iteration {self.kmax}")
        return self.best_solution

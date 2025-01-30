import numpy as np
import copy
from .Optimizer import Optimizer
from .Individual import Individual

class SimulatedAnnealingOptimizer(Optimizer):
    """Simulated Annealing optimizer for DNA structural circularity. Subclass of Optimizer."""

    def __init__(self, generations=100, population_size=50, mutation_rate=0.1, kmax=10000, emax=1, initial_temp=100):
        """Initialize the simulated annealing optimizer with population size and mutation rate. Uses the constructor of the parent class."""

        super().__init__(generations, population_size, mutation_rate)
        self.kmax = kmax                    # Maximum number of iterations
        self.emax = emax                    # Acceptable score threshold to stop the algorithm
        self.initial_temp = initial_temp    # Initial temperature for simulated annealing
        self.load_table()                   # Load the reference table
        self.best_distance = float('inf')   # Keep track of the best distance found

    def mutate(self, individual: Individual):
        """Apply random mutations to an individual. Returns the mutated individual."""

        mutated = False                 # Track if any mutation occurred
        table = individual.getTable()   # Access the individual's rotation table
        
        while not mutated:              # Ensure at least one mutation occurs
            for dinucleotide in table:
                dinu = table[dinucleotide]
                if np.random.random() < self.getMutationRate():
                    individual.setTwist(dinucleotide, np.random.uniform(dinu[0]-dinu[3], dinu[0]+dinu[3]))
                    mutated = True
                if np.random.random() < self.getMutationRate():
                    individual.setWedge(dinucleotide, np.random.uniform(dinu[1]-dinu[4], dinu[1]+dinu[4]))
                    mutated = True
                if np.random.random() < self.getMutationRate():
                    individual.setDirection(dinucleotide, np.random.uniform(dinu[2]-dinu[5], dinu[2]+dinu[5]))
                    mutated = True
        
        return individual

    def optimize(self, dna_sequence:str):
        """Run the simulated annealing optimization algorithm. Returns the best solution found."""

        # Initialisation
        current_individual = copy.deepcopy(self.getIndRef())
        current_fitness, current_distance = self.calculate_fitness(current_individual, dna_sequence)
        self.setBestSolution(copy.deepcopy(current_individual))
        self.setBestFitness(current_fitness)
        self.best_distance = current_distance
        print(f"Initial fitness and distance: {current_fitness:.2f}, {current_distance:.2f}")
        k = 0

        while k < self.kmax and current_fitness > self.emax:
            # Generate a neighbor of the current solution by mutation
            neighbor_individual = self.mutate(copy.deepcopy(current_individual))

            # Calculate the fitness of the neighbor
            neighbor_fitness, neighbor_distance = self.calculate_fitness(neighbor_individual, dna_sequence)

            # Accept the neighbor if it has a better fitness or with a certain probability
            temperature = self.initial_temp * (1 - k / self.kmax)   # Decrease the temperature over time, linearly in this case
            if neighbor_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - neighbor_fitness) / temperature):
                current_individual = neighbor_individual
                current_fitness = neighbor_fitness
                current_distance = neighbor_distance

            # Keep track of the best solution found
            if neighbor_fitness < self.getBestFitness():
                self.setBestSolution(copy.deepcopy(neighbor_individual))
                self.setBestFitness(neighbor_fitness)
                self.best_distance = neighbor_distance

            if k % 100 == 0:
                print(f"Best fitness and distance: {self.getBestFitness():.2f} and {self.best_distance:.2f} for iteration {k}")

            
            k += 1  # Increment the iteration counter    

        print(f"Best fitness and distance: {self.getBestFitness():.2f} and {self.best_distance:.2f} for iteration {self.kmax}")
        return self.getBestSolution()

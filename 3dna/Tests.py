import unittest
import random
import math
import copy
import numpy as np
import os

from Individual import Individual
from genetic_optimizer import GeneticOptimizer


#################################
# 1) Test de la fitness
#################################

class TestCalculateFitness(unittest.TestCase):
    def test_calculate_fitness(self):
        """
        Vérifie que la valeur de fitness calculée est > 0
        """
        # Create test optimizer
        optimizer = GeneticOptimizer()
        
        # Test data with valid sequences from table.json
        sequence = "ATCG"  # Valid sequence that exists in table.json
        fitness_value = optimizer.calculate_fitness(optimizer.ind_ref, sequence)[0]  # Get only the fitness value
        
        # Verify
        self.assertGreater(fitness_value, 0, "La valeur de fitness doit être positive.")


#################################
# 2) Test du crossover
#################################

class TestCrossover(unittest.TestCase):
    def test_crossover(self):
        """
        Vérifie que la méthode 'crossover' génère bien un enfant
        ayant la même taille que ses parents.
        """
        optimizer = GeneticOptimizer()
        
        # Create parent individuals
        parent1 = Individual()
        parent2 = Individual()

        # Perform crossover
        child = optimizer.crossover(parent1, parent2)
        
        # Verify child has same table size as parents
        self.assertEqual(
            len(child.getTable()),
            len(parent1.getTable()),
            "L'enfant doit avoir le même nombre d'entrées que ses parents"
        )


#################################
# 3) Tests sur create_new_gen
#################################

class TestCreateNewGen(unittest.TestCase):
    def setUp(self):
        """
        Initialisation commune à tous les tests
        """
        self.optimizer = GeneticOptimizer()
        self.population = []
        # Create population_size individuals
        for i in range(self.optimizer.population_size):
            ind = Individual()
            ind.setFitness(float(i))
            self.population.append(ind)

    def test_selection_best_random(self):
        """Test avec type_choosing_parent='best' et type_matching='random'"""
        sequence = "ATCG"
        population = []
        for _ in range(self.optimizer.population_size):
            ind = self.optimizer.create_individual()
            fitness = self.optimizer.calculate_fitness(ind, sequence)[0]  # Get only the fitness value
            ind.setFitness(fitness)
            ind.setCalculated(True)
            population.append(ind)

        new_gen = self.optimizer.create_new_gen(
            population=population,
            gen=0,  # Add gen parameter
            type_choosing_parent="best",
            type_matching="random"
        )
        self.assertEqual(len(new_gen), self.optimizer.population_size)

    def test_selection_par_rang_random(self):
        """Test avec type_choosing_parent='selection par rang'"""
        sequence = "ATCG"
        population = []
        for _ in range(self.optimizer.population_size):
            ind = self.optimizer.create_individual()
            fitness = self.optimizer.calculate_fitness(ind, sequence)[0]  # Get only the fitness value
            ind.setFitness(fitness)
            ind.setCalculated(True)
            population.append(ind)

        new_gen = self.optimizer.create_new_gen(
            population=population,
            gen=0,  # Add gen parameter
            type_choosing_parent="selection par rang",
            type_matching="random"
        )
        self.assertEqual(len(new_gen), self.optimizer.population_size)

    def test_selection_tournoi_random(self):
        """Test avec type_choosing_parent='tournoi'"""
        sequence = "ATCG"
        population = []
        for _ in range(self.optimizer.population_size):
            ind = self.optimizer.create_individual()
            fitness = self.optimizer.calculate_fitness(ind, sequence)[0]  # Get only the fitness value
            ind.setFitness(fitness)
            ind.setCalculated(True)
            population.append(ind)

        new_gen = self.optimizer.create_new_gen(
            population=population,
            gen=0,  # Add gen parameter
            type_choosing_parent="tournoi",
            type_matching="random"
        )
        self.assertEqual(len(new_gen), self.optimizer.population_size)

    def test_selection_roulette_tournament(self):
        """Test avec type_choosing_parent='selection par roulette' et type_matching='tournament'"""
        sequence = "ATCG"
        population = []
        for _ in range(self.optimizer.population_size):
            ind = self.optimizer.create_individual()
            fitness = self.optimizer.calculate_fitness(ind, sequence)[0]  # Get only the fitness value
            ind.setFitness(fitness)
            ind.setCalculated(True)
            population.append(ind)

        new_gen = self.optimizer.create_new_gen(
            population=population,
            gen=0,  # Add gen parameter
            type_choosing_parent="selection par roulette",
            type_matching="tournament"
        )
        self.assertEqual(len(new_gen), self.optimizer.population_size)

    def test_selection_best_meritocratie(self):
        """Test avec type_choosing_parent='best' et type_matching='meritocratie'"""
        sequence = "ATCG"
        population = []
        for _ in range(self.optimizer.population_size):
            ind = self.optimizer.create_individual()
            fitness = self.optimizer.calculate_fitness(ind, sequence)[0]  # Get only the fitness value
            ind.setFitness(fitness)
            ind.setCalculated(True)
            population.append(ind)

        new_gen = self.optimizer.create_new_gen(
            population=population,
            gen=0,  # Add gen parameter
            type_choosing_parent="best",
            type_matching="meritocratie"
        )
        # Vérifie que la taille de la nouvelle génération est raisonnable
        # La méthode méritocratie peut générer une population légèrement plus grande
        self.assertGreater(len(new_gen), 0)
        self.assertLess(len(new_gen), self.optimizer.population_size * 2)  # Ne devrait pas être plus du double

    def test_optimize(self):
        """Test de la méthode optimize"""
        sequence = "AATT"  # Séquence valide
        generations = 2  # Petit nombre pour le test
        result = self.optimizer.optimize(sequence, generations)
        
        # Vérifie que le résultat est un Individual
        self.assertIsInstance(result, Individual)
        
        # Vérifie que le meilleur individu a une fitness
        self.assertIsNotNone(result.getFitness())
        self.assertGreater(result.getFitness(), 0)


if __name__ == '__main__':
    unittest.main()
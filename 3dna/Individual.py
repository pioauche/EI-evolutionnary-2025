from decimal import Rounded
from .RotTable import RotTable

class Individual(RotTable):
    def __init__(self, filename: str=None):
        super().__init__(filename)
        self.fitness = 0
        self.calculated=False
    def setFitness(self, fitness: float):
        self.fitness = fitness
    def getFitness(self):
        return self.fitness
    def isCalculated(self):
        return self.calculated
    def calculate(self,b):
        self.calculated=b

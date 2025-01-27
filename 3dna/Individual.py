from decimal import Rounded
from .RotTable import RotTable

class Individual(RotTable):
    def __init__(self, filename: str=None):
        super().__init__(filename)
        self.fitness = 0
        self.calculated=False
        self.pair= 16
        
    def setFitness(self, fitness: float):
        self.fitness = fitness
        self.calculated=True

    def getFitness(self):
        return self.fitness

    def setCalculated(self, calculated: bool):
        self.calculated = calculated

    def isCalculated(self):
        return self.calculated

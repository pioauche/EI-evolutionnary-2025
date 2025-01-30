from RotTable import RotTable

class Individual(RotTable):
    def __init__(self, filename: str=None):
        super().__init__(filename)
        self.__fitness = 0
        self.__calculated=False
        self.__pair= 16
        
    def setFitness(self, fitness: float):
        self.__fitness = fitness
        self.__calculated=True

    def getFitness(self):
        return self.__fitness

    def setCalculated(self, calculated: bool):
        self.__calculated = calculated

    def isCalculated(self):
        return self.__calculated
    def calculate(self,b):
        self.__calculated=b

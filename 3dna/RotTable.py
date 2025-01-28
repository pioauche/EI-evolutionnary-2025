from json import load as json_load
from os import path as os_path

here = os_path.abspath(os_path.dirname(__file__))

class RotTable:
    """Represents a rotation table"""

    # 3 first values: 3 angle values
    # 3 last values: SD values

    def __init__(self, filename: str = None):
        if filename is None:
            filename = os_path.join(here, 'table copy.json')
        self.__rot_table = json_load(open(filename))
        
    ###################
    # WRITING METHODS #
    ###################
    def setTwist(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][0] = value

    def setWedge(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][1] = value

    def setDirection(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][2] = value
    
    def addTwist(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][0] += value

    def addWedge(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][1] += value

    def addDirection(self, dinucleotide: str, value: float):
        self.__rot_table[dinucleotide][2] += value
    
    ###################
    # READING METHODS #
    ###################
    def getTwist(self, dinucleotide: str) -> float:
        return self.getTable()[dinucleotide][0]

    def getWedge(self, dinucleotide: str) -> float:
        return self.getTable()[dinucleotide][1]

    def getDirection(self, dinucleotide: str) -> float:
        return self.getTable()[dinucleotide][2]

    def setTable(self, table: dict):
        self.__rot_table = table

    def getTable(self) -> dict:
        return self.__rot_table

    ###################

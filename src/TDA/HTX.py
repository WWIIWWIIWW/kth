import cantera as ct
import numpy as np

class HTX:
    def __init__(self, fluid):
        self.gas = fluid

    def TPX(self, Tin, Pin, X):
        self.gas.TPX = Tin, Pin, X
        self.gas.equilibrate("TP")
        
        #Wrap gas in Quantity to access get_deltaN()
        self.gas = ct.Quantity(self.gas, moles = 1)  
        
    def TPY(self, Tin, Pin, Y):
        self.gas.TPY = Tin, Pin, Y
        self.gas.equilibrate("TP")
        
        #Wrap gas in Quantity to access get_deltaN()
        self.gas = ct.Quantity(self.gas, moles = 1) 
        
    def arg_name(arg):
        return inspect.getargspec(arg)[0]
        
    def printState(self, n):
        print('\n***************** State {0} ******************'.format(n))
        print(self.gas.report())

    def heat_to(self, arg1, Pout, HTX_type = "TP"):
        """
        heat exchange using TPH
        """
       
        self.h0 = self.gas.h
        self.s0 = self.gas.s
        if type(self.gas).__name__ == "Quantity":
            self.n0 = self.gas.moles
            
        if HTX_type == "TP":
            self.gas.TP = arg1, Pout
            self.gas.equilibrate("TP")
        elif HTX_type == "HP":
            self.gas.HP = arg1, Pout
            self.gas.equilibrate("HP")
            
        self.h1 = self.gas.h
        self.s1 = self.gas.s
        if type(self.gas).__name__ == "Quantity":
            self.n1 = self.gas.moles

        self.work = self.h1 - self.h0
        self.T    = self.gas.T
        self.deltaH = self.h1 - self.h0
        self.deltaS = self.s1 - self.s0
        
        
    def get_work(self):
        return self.work
        
    def get_T(self):
        return self.T
        
    def get_deltaH(self):
        return self.deltaH
        
    def get_deltaS(self):
        return self.deltaS

    def get_deltaN(self):
        if type(self.gas).__name__ == "Quantity":
            deltaN = self.n1 - self.n0
        else:
            deltaN = "Warning: Unable to export mole changes as basis does not contain 'Quantity'!"
        return deltaN
        

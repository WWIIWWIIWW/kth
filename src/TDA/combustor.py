import cantera as ct
import numpy as np
class combustor:
    def __init__(self, fluid):
        self.gas = fluid

    def init_fluid(self, Tin, Pin):
        self.gas.TP = Tin, Pin
        self.gas.equilibrate("TP")
        
        #Wrap gas in Quantity to access get_deltaN()
        self.gas = ct.Quantity(self.gas, moles = 1) 

    def printState(self, n):
        print('\n***************** State {0} ******************'.format(n))
        print(self.gas.report())

    def burn_to(self, Tout):
        """
        combustor burn fuel to temperature Tout at constant inlet pressure
        """
        
        self.h0 = self.gas.h
        self.s0 = self.gas.s
        
        if type(self.gas).__name__ == "Quantity":
            self.n0 = self.gas.moles
            
        self.gas.TP = Tout, None
        self.gas.equilibrate("TP")
            
        self.h1 = self.gas.h
        self.s1 = self.gas.s
        if type(self.gas).__name__ == "Quantity":
            self.n1 = self.gas.moles
            
        self.work = self.h1 - self.h0
        self.T    = self.gas.T
        
    def get_work(self):
        return self.work
        
    def get_T(self):
        return self.T
        
    def get_deltaH(self):
        return self.h1 - self.h0
        
    def get_deltaS(self):
        return self.s1 - self.s0

    def get_deltaN(self):
        if type(self.gas).__name__ == "Quantity":
            deltaN = self.n1 - self.n0
        else:
            deltaN = "Warning: Unable to export mole changes as basis does not contain 'Quantity'!"
        return deltaN
        

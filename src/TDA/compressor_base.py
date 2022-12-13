import cantera as ct
import numpy as npz

class compressor_base:
    def __init__(self, fluid, efficiency):
        self.gas = fluid
        self.eta = efficiency
        
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

    def printState(self, n):
        print('\n***************** State {0} ******************'.format(n))
        print(self.gas.report())

    def compress_to(self, Pout):
        """
        Adiabatically compress a fluid to pressure p_final
        , using a compressor with isentropic efficiency eta.
        """

        self.h0 = self.gas.h
        self.s0 = self.gas.s
        
        if type(self.gas).__name__ == "Quantity":
            self.n0 = self.gas.moles
            
        self.gas.SP = self.s0, Pout
        h1s = self.gas.h
        isentropic_work = h1s - self.h0 
        
        #calculate work into system
        self.work = isentropic_work * self.eta
        
        #calculate enthalpy at exit of compressor
        self.h1 = self.work + self.h0 
        
        #set equilibriate status at exit of compressor
        self.gas.HP = self.h1, Pout
        self.gas.equilibrate("HP")
        self.s1 = self.gas.s

        if type(self.gas).__name__ == "Quantity":
            self.n1 = self.gas.moles
            
        self.T    = self.gas.T
        self.deltaH = self.h1 - self.h0
        self.deltaS = self.s1 - self.s0

    def get_T(self):
        return self.T
    #below functions need to be overwritten in staged calculation
    def get_work(self):
        return self.work
              
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
        
    #def show_PT_diagram(self):
class loop_compressor(compressor_base):

    def __init__(self, fluid, efficiency):
        super().__init__(fluid, efficiency)
        self.solution = []

    def init_fluid(self, Tin_array, Pin_array, X):
        self.Tin_array = Tin_array
        self.Pin_array = Pin_array
        self.X = X
  
    def compress_to(self, Pout):

        for Tin in self.Tin_array:
            for Pin in self.Pin_array:
                comp = compressor_base(fluid = self.gas, efficiency = self.eta)
                comp.init_fluid(Tin = Tin, Pin = Pin, X = self.X)
                comp.compress_to(Pout = Pout)

                Work = comp.get_work()
                T_out = comp.get_T()
                deltaH = comp.get_deltaH()
                deltaS = comp.get_deltaS()
                deltaN = comp.get_deltaN()

                self.solution.append((Tin, Pin, T_out, Work, deltaH, deltaS, deltaN))
                
    def save(self, path = "./"):
        np.savetxt(path + type(self).__name__ + "_solution.txt", self.solution)
        
        

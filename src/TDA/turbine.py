import cantera as ct
import numpy as np
from .turbine_base import turbine_base

class turbine(turbine_base):
    #constructor default number of stages = 1
    def __init__(self, fluid, efficiency, nStages = 1):
            # invoking the __init__ of the parent class
            super().__init__(fluid, efficiency)
            self.nStages = nStages

    def get_divider(self, Pout):
        """
        Multiplier for decreasing pressure from Phigh to Plow based on nStages
        """
        divider = np.power(self.gas.P/Pout,(1/self.nStages))
        
        return divider

    def stage_expand_to(self, Pout, show = False):
        divider = self.get_divider(Pout) #use divider for expand_to
        Phigh = self.gas.P  # use Phigh for expand_to
        self.staged_work = 0
        
        for stage in range(self.nStages): #0,1,2,3,4 if nStages = 5
            super().expand_to(Pout = Phigh / divider**(stage+1))
 
            self.staged_work += super().get_work() 
            if show:
                print ("Turbine in = {}Pa, out = {}Pa, Work required at Stage {} = {}J.".format(Phigh * divider**(stage), Phigh * divider**(stage+1), stage,  super().get_work()))
            
        if show:
            print ("Turbine staged work = {}J".format(self.staged_work))

    def get_staged_work(self):
        return self.staged_work

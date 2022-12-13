import cantera as ct
import numpy as np
from .compressor_base import compressor_base

class compressor(compressor_base):
    #constructor default number of stages = 1
    def __init__(self, fluid, efficiency, nStages = 1):
            # invoking the __init__ of the parent class
            super().__init__(fluid, efficiency)
            self.nStages = nStages

    def get_multiplier(self, Pout):
        """
        Multiplier for increasing pressure from Plow to Phigh based on nStages
        """
        multiplier = np.power(Pout/self.gas.P,(1/self.nStages))
        
        return multiplier
        
    def stage_compress_to(self, Pout, show = False):
        multiplier = self.get_multiplier(Pout) #use multiplier for compress_to
        Plow = self.gas.P  # use Plow for compress_to
        self.staged_work = 0
        
        for stage in range(self.nStages): #0,1,2,3,4 if nStages = 5
            super().compress_to(Pout = Plow * multiplier**(stage+1))
 
            self.staged_work += super().get_work() 
            if show:
                print ("Compressor in = {}Pa, out = {}Pa, Work required at Stage {} = {}J.".format(Plow * multiplier**(stage), Plow * multiplier**(stage+1), stage,  super().get_work()))
            
        if show:
            print ("Compressor staged work = {}J".format(self.staged_work))
            
    def get_staged_work(self):
        return self.staged_work

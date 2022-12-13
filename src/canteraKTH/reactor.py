import cantera as ct
import pandas as pd
from typing import Sequence, TypedDict, NewType, Tuple

qty   = ct.composite.Quantity
sol   = ct.composite.Solution
rtr   = ct._cantera.Reactor
def mix(X_A: TypedDict, X_B: TypedDict, mech: str, thermo: str = 'HP',
       printReport: bool = False, T: float = 303.15, P: float = 101325) -> qty:
    """
    Mixing A with B in combustor, and export mixture as C.
    Conditions:
    T： Temperature, P： Pressure
    X_A： mole fraction of A
    X_B： mole fraction of A
     
    """
    gas = ct.Solution(mech)

    nmole = 1.0
    A = set_inlet_stream(gas, X_A, nmole, thermo, printReport, T, P)

    # Set the molar flow rates corresponding to stoichiometric reaction,
    # CH4 + 2 O2 -> CO2 + 2 H2O
    nO2 = A.X[A.species_index('O2')]
    nmole = nO2 * 0.5
    B = set_inlet_stream(gas, X_B, nmole, thermo, printReport, T, P)

    # Compute the mixed state
    C = A + B
    if printReport:
        print(C.report())
    return C
    
def set_inlet_stream(gas: sol, X: TypedDict, nmole: float = 1.0, thermo: str = 'HP',
                    printReport: bool = False, T: float = 303.15, 
                    P: float = 101325) -> qty:
             
    inlet_stream = ct.Quantity(gas, constant = thermo)
    inlet_stream.TPX = T, P, X
    inlet_stream.moles = nmole
    
    if printReport:
        print(inlet_stream.report())
        
    return inlet_stream
    
def auto_ignite(mixture: pd.DataFrame, mech: str = 'gri30.yaml', 
                limT: float = 3000) -> Sequence[float]:
    """
    Calculate auto-ignition temperatire of mixture using mech. (reactor method)
    limT: if T loop goes beyond limit, mixture is assumed non-ignitable.
    """
    
    if type(mixture) == pd.DataFrame:
        X = mixture.to_dict("index")
    else:
        X = [mixture]
        
    T_ign = []
    for idx in range(len(X)):
        gas = ct.Solution(mech)

        r, T  = reactor(gas, X[idx])

        while (r.T - T < 50):
        
            T += 5
            r, T  = reactor(gas, X[idx], T)
            
            if T > limT:
                break
               
        T_ign.append(T)
        print ('Auto-igniton temperature = {}K'.format(T))
        
    return T_ign
    
def auto_ignite_CRN(mixture: pd.DataFrame, mech: str = 'gri30.yaml',
                    limT: float = 3000) -> Sequence[float]:
    """
    Calculate auto-ignition temperature of mixture using mech. (CRN method)
    limT: if T loop goes beyond limit, mixture is assumed non-ignitable.
    """
    if type(mixture) == pd.DataFrame:
        X = mixture.to_dict("index")
    else:
        X = [mixture]
        
    T_ign = []
    for idx in range(len(X)):
        gas = ct.Solution(mech)

        r, T  = CRN(gas, X[idx])

        while r.T - T < 50:
        
            T += 5
            r, T  = CRN(gas, X[idx], T)
            
            if T > limT:
                break
               
        T_ign.append(T)
        print ('Auto-igniton temperature = {}K'.format(T))
        
    return T_ign
    
def reactor(gas: sol, X: dict, T: float = 600.00, P: float = 101325) -> Tuple[rtr, float]:
    """
    Initialize reactor with gas, X
    Start calculating ignition temperature from T = 600K, then increment
    """

    residence_time = 1 #s
    
    gas.TPX = T, P, X
    
    r  = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    sim.advance(residence_time)
    
    return (r, T)
    
def CRN(gas: sol, X: dict, T: float = 600.00, P: float = 101325) -> Tuple[rtr, float]:
    """
    Initialize CRN with gas, X
    Start calculating ignition temperature from T = 600K, then increment
    """
    residence_time = 1 #s
    
    gas.TPX = T, P, X
    
    t1 = ct.Reservoir(contents = gas, name = 'inlet')     #tank1/inlet
    t2 = ct.Reservoir(contents = gas, name = 'exhaust')   #tank2/exhaust

    r = ct.IdealGasReactor(contents = gas, name = 'PSR', energy='on')
    
    def mdot_inlet(t):
        """
        Use a variable mass flow rate to keep the residence time in the reactor
        constant (residence_time = mass / mass_flow_rate). The mass flow rate 
        function can access variables defined in the calling scope, including
        state variables of the Reactor object (combustor) itself.
        """
        return (r.mass / residence_time)

    inlet_to_PSR   = ct.MassFlowController(t1, r, mdot=mdot_inlet)
    PSR_to_exhaust = ct.PressureController(r, t2, master = inlet_to_PSR, K=0.01)
    sim = ct.ReactorNet([r])
    sim.advance_to_steady_state()
    
    return (r, T)
   

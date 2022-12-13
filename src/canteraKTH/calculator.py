import cantera as ct
import pandas as pd
import numpy as np
from typing import Sequence, TypedDict, NewType, Tuple

sol   = ct.composite.Solution
##None cantera-related functions:##
def calSurfaceAvg(mesh, species: str = "", path: str = "") -> float:
    """
    Calculate area-weighted emission at a vtk plane sampled in OpenFOAM.
    """
    """
    Sample usage: (if OF env is sourced)
    path = '$FOAM_CASE/postProcessing/planetoAvg/0.2/NOMean_xNormal_1.vtk'
    df, mesh = ckth.import_vtk_data(path)
    val = ckth.calSurfaceAvg(mesh, spec, path)
    """
    
    cell_info = mesh.ptc().compute_cell_sizes()
    area      = cell_info.area
    area_C    = cell_info.cell_arrays["Area"]
    val_C     = cell_info.get_array(species)
    
    avg = sum(val_C * area_C) / area
    
    return avg
    
##cantera-related functions:##
def calcLewis(mixture: pd.DataFrame, mech: str = 'gri30.yaml', T: float = 303.15, P: float = 101325) -> pd.DataFrame:
    """
    Calculate Lewis number at flame front.
    """
    if type(mixture) == pd.DataFrame:
        X = mixture.to_dict("index")
    else:
        X = [mixture]
        
    data = []
    for idx in range(len(X)):
        gas = ct.Solution(mech)
        f   = calcFlame(gas, X[idx], T, P)
        
        gas.TPX = get_flame_front_TPX(f) #re-define gas with flame front information.
        
        viscosity   = gas.viscosity
        density     = gas.density_mass
        conductivity= gas.thermal_conductivity #W/m-K
        Cp          = gas.cp_mass
        
        Le = []
        for species in gas.species_names:

            mass_diff = gas.mix_diff_coeffs_mass[gas.species_index(species)]
            thermal_diff = conductivity/Cp/density
            Le.append(thermal_diff/mass_diff)
            
        data.append(Le) 
        df_Le = pd.DataFrame(data, columns = gas.species_names)
    
    return df_Le

def calcSL(mixture: pd.DataFrame, mech: str = 'gri30.yaml', T: float = 303.15, P: float = 101325) -> pd.DataFrame:
    """
    Calculate flame speed.
    """
    if type(mixture) == pd.DataFrame:
        X = mixture.to_dict("index")
    else:
        X = [mixture]
        
    SL = []
    for idx in range(len(X)):
        gas = ct.Solution(mech)
        f   = flame(gas, X[idx], T, P)
        print ("Flame speed = {m/s}".format(f.u[0]))
        SL.append(f.u[0])
        
    return pd.DataFrame(SL, columns = ['SL'])

def calcFlame(gas: sol, X: dict, T: float = 303.15, P: float = 101325):
    """
    Initialize oneD flame calculation with gas, X.
    return the flame.
    """
    ##############
    Lx=0.02
    tol_ss      = [1.0e-6, 1.0e-14]        # [rtol atol] for steady-state problem
    tol_ts      = [1.0e-5, 1.0e-13]        # [rtol atol] for time stepping
    loglevel    = 0                        # amount of diagnostic output (0
    refine_grid = True                     # True to enable refinement
    ##############

    gas.TPX = T, P, X
    print ("Equivalence ratio = {}".format(gas.get_equivalence_ratio()))
        
    f = ct.FreeFlame(gas, width=Lx)
    f.transport_model = 'Multi'
    f.soret_enabled=True
        
    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_transient_tolerances(default=tol_ts)
    f.set_refine_criteria(ratio=3, slope=0.01, curve=0.01)

    print ('Solving flame ....')
    f.solve(loglevel=loglevel, refine_grid=refine_grid, auto=True)
    
    return f

def get_flame_front_idx(f):
    """
    Get the position (idx) in oneD flame where temperature gradient is maximum.
    """
    
    x = f.grid
    T = f.T
    Dx = np.diff(x)
    DT =np.diff(T)

    ff_idx = np.where(np.abs(DT/Dx) == np.amax(np.abs(DT/Dx)))
    
    return ff_idx[0][0]

def get_flame_front_TPX(f) -> TypedDict:
    """
    Get flame front TPX for a oneD flame where temperature gradient is maximum.
    """
    
    idx = get_flame_front_idx(f)

    X   = f.X[:][:,idx]
    T   = f.T[idx]
    P   = f.P

    print ("Flame front temperature = {}K".format(T))
    
    return T, P, X

def get_thermal_thickness(f):
    """
    Get the thermal thickness of oneD flame.
    """
    
    x=f.grid
    T=f.T
    Dx=np.diff(x)
    DT=np.diff(T)
    
    return (np.amax(T)-np.amin(T))/np.amax(np.abs(DT/Dx))
    
def get_progress_variable(f):
    """
    Get the progress variable of oneD flame.
    """
    
    return (f.T - f.T[0]) / (f.T[-1] - f.T[0])
    

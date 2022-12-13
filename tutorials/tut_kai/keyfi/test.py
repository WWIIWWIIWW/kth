import os
import keyfi as kf

path_input  = './input/2DMRB.vtk'
data, mesh = kf.import_vtk_data(path_input)

#cleaned_data = kf.clean_data(data)

cleaned_data = kf.clean_data(data, dim=2, vars_to_drop=None, 
vars_to_keep=['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'CH2', 'CH2S', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH2OH', 'N2']
)

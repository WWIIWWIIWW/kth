import numpy as np
import pandas as pd
import pyvista as pv
import csv
import os

from typing import Sequence, Type, TypedDict
from .calculator import get_progress_variable

def import_vtk_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a vtk data file.
    Also returns mesh pyvista object.
    '''
    if not path:
        path = input('Enter the path of your vtk data file: ')

    mesh = pv.read(path)

    # Remove vtkGhostType automatically by applying threshold, replace step in paraview.
    if 'vtkGhostType' in mesh.array_names:
        mesh = mesh.threshold(value = [0,0.99], scalars = "vtkGhostType",
               invert=False, continuous = True, preference = "cell",
               all_scalars = False)

    vector_names = []

    # Detect which variables are vectors
    for var_name in mesh.array_names:
        if np.size(mesh.get_array(var_name)) != mesh.n_points:
            vector_names.append(var_name)

    # Make a dataframe from only scalar mesh arrays (i.e. exclude vectors)
    var_names = [name for name in mesh.array_names if name not in vector_names]
    var_arrays = np.transpose([mesh.get_array(var_name) for var_name in var_names])
    df = pd.DataFrame(var_arrays, columns=var_names)

    # Add the vectors back with one row per component
    for vector_name in vector_names:
        # Get dimension of data e.g., 1D or 2D
        data_dim = mesh.get_array(vector_name).ndim

        if data_dim == 1: 
            pass
        else:
            # Get dimension (number of columns) of typical vector
            dim = mesh.get_array(vector_name).shape[1]
            # split data using dim insteady of hard coding
            df[[vector_name + ':' + str(i) for i in range(dim)]] = mesh.get_array(vector_name)

    return df, mesh

def export_vtk_data(mesh: Type, path: str = '', newData: np.ndarray = None,
                    newData_name: str = 'newData') -> None:
    '''
    Exports vtk file with mesh. If cluster labels are passed it
    will include them in a new variable
    '''
    if newData is not None:
        mesh[newData_name] = newData
    mesh.save(path)

def clean_data(data: pd.DataFrame, dim: int = 2, vars_to_drop: Sequence[str] = None,
               vars_to_keep: Sequence[str] = None) -> pd.DataFrame:
    '''    
    Removes ghost cells (if present) and other data columns that
    are not relevant for the dimensionality reduction (i.e. spatial 
    coordinates) from the original data.
    '''
    if dim not in [2, 3]:
        raise ValueError(
            'dim can only be 2 or 3. Use 2 for 2D-plane data and 3 for 3D-volume data')

    cols_to_drop = []

    if 'Points:0' in data.columns:
        cols_to_drop.append(['Points:0', 'Points:1', 'Points:2'])

    if 'vtkGhostType' in data.columns:
        data.drop(data[data.vtkGhostType == 2].index, inplace=True)
        cols_to_drop.append('vtkGhostType')

    if vars_to_keep is not None:
        # Return cleaned data based on preferred var
        cleaned_data = data[["{}".format(var) for var in vars_to_keep]]
    else:
        # drop undesired variables based on 'dim' and 'var_to_drop'
        if 'U:0' in data.columns and dim == 2:
            cols_to_drop.append('U:2')

        if vars_to_drop is not None:
            cols_to_drop.extend(vars_to_drop)

        cleaned_data = data.drop(columns=cols_to_drop, axis=1)
        cleaned_data.reset_index(drop=True, inplace=True)

    return cleaned_data

def merge_csv_data(read_path: str = None, merged_name: str = None, merged_path: str = None, save: bool = False) -> pd.DataFrame:
    '''    
    Merge data in typical directory, labled with file name without extension.
    e.g.:
    /home/output/flames/flame1.csv
    /home/output/flames/flame2.csv
            col1    col2    label
    0                       flame1
    1                       flame1
    2                       flame2
    3                       flame2
    '''
    # default merged_path = read_path
    merged_path = read_path if merged_path is None else merged_path
    
    filelist = [f.split('.')[0] for f in os.listdir(read_path) if not f.startswith(merged_name)]
    
    df_list = []
    for item in filelist:

        df = pd.read_csv(read_path + item + '.csv')
        labeled_df = df.assign(label = '{}'.format(item))
        df_list.append(labeled_df)
    df = pd.concat(df_list)
    
    print('Read data merged. \n')
    
    if save:
        _save_merged_data(df, merged_name, merged_path)
        
    return df

def _save_merged_data(merged_data: pd.DataFrame, merged_name: str = None, merged_path: str = None) -> None:
    
    merged_data.to_csv(merged_path + merged_name + '.csv')
    
    print('Merged data saved. \n')

def getList(new_data):
    key   = new_data.keys()
    value = new_data.values()
    return list(key), list(value)

def write_flame(f, gas, idx, fpath: str = None, new_Data = TypedDict) -> None:
    '''    
    write flame csv to output/flames with new data added.
    '''

    fname = fpath + '/flame{}.csv'.format(idx+1)
    
    key, value = getList(new_Data)

    z = f.flame.grid
    T = f.T
    u = f.u
   
    with open(fname, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['z (m)', 'u (m/s)', 'T (K)', 'rho (kg/m3)'] +
                list(gas.species_names) + key)

        for n in range(f.flame.n_points):
            f.set_gas_state(n)
            writer.writerow([z[n], u[n], T[n], gas.density] + list(gas.X) + [item[n] for item in value])
            
    print('Output case {} written to {}'.format(idx+1, fname))

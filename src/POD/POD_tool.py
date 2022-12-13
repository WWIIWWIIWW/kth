import os
import numpy as np
import glob
import pandas as pd
import pyvista as pv

import matplotlib.pyplot as plt
from matplotlib import lines

from typing import Union, Sequence, Tuple, Type
# from algorithm import *
import time as clock
from numpy import linalg as la

plt.rc('xtick', labelsize='16')
plt.rc('ytick', labelsize='16')
plt.rc('font', family='serif')
plt.rc('text', usetex=False)
plt.rc('legend', fontsize=11)
plt.rcParams.update({'figure.max_open_warning': 0})

"""
m_samples
n_features
l_snapshots
"""


def get_snapshots_time(snapshots_dir, savepath: str = '', savename: str = '', save: bool = False) -> list:
    """
    Collect number of snapshots from snapshots_dir, return array containing it.
    """

    print("Getting snapshots....")

    snapshot_list = [float(i) for i in os.listdir(snapshots_dir)]
    # snapshot_list = [int(num) if float(num).is_integer() else float(num) for num in os.listdir(snapshots_dir)]
    snapshot_arr = np.sort(np.asarray(snapshot_list))

    snapshot_list = [int(x) if x == int(x) else x for x in snapshot_arr]

    #print(snapshot_arr)

    if save:
        os.makedirs(savepath, exist_ok=True)
        np.savetxt(savepath + savename + '.txt', snapshot_arr)

    return snapshot_list


def update_path(path: str = "", var_name: str = ""):
    """
    Look for file folder with name containing the var_name.
    Return new_path to read this folder.
    """
    # #This block was trying to solve the issue of 1.0 and 1 problem in directory, simply solution has been employed.
    # try:
    #     newpath = os.listdir(path)  # dirlist
    # except:
    #     splitpath = path.split('/')
    #     splitpath[-2] = str(int(float(splitpath[-2])))
    #     newpath = ('/').join(splitpath)

    newpath = path

    for string in os.listdir(newpath):  # dirlist
        if var_name in string:
            substring = string

    return (newpath + substring)


def get_dimension_mesh(snapshots_dir: str = '', snapshot_time: np.ndarray = None, var_name: str = '') -> np.ndarray:
    """
    Read a single snapshot to get m_samples, n_features, and l_snapshots as array.
    """

    if isinstance(snapshot_time, np.ndarray):
        print ("snapshot_time is an array")
        snapshot_time = [int(x) if x == int(x) else x for x in snapshot_time]

    path = snapshots_dir + str(snapshot_time[0]) + "/"  # use str for concatenate directory
    new_path = update_path(path, var_name)

    df, mesh = import_vtk_data(path=new_path, var_name=var_name)

    m_samples = df.shape[0]
    n_features = df.shape[1]
    l_snapshots = len(snapshot_time)

    return (np.array([m_samples, n_features, l_snapshots]), mesh)


def import_vtk_data(path: str = '', var_name: str = '') -> pd.DataFrame:
    """
    Creates a pandas dataframe [samples, nfeatures] from path.
    Also returns mesh pyvista object.
    """

    if not path:
        path = input('Enter the path of where you save your snapshots: ')
    if not var_name:
        var_name = input('Enter the name of your variable: ')

    mesh = pv.read(path)

    var_array = mesh.get_array(var_name, preference='cell')

    data_dim = var_array.ndim

    if data_dim == 1:
        df = pd.DataFrame(var_array, columns=[var_name])
    else:
        # Get dimension (number of columns) of typical vector
        dim = var_array.shape[1]
        # split data using dim insteady of hard coding
        df = pd.DataFrame(var_array, columns=[var_name + ':' + str(i) for i in range(dim)])
        # df = pd.DataFrame()
        # df[[var_name + ':' + str(i) for i in range(dim)]] = var_array

    return df, mesh


def transform_2D_matrix_to_3D(POD_mode_matrix, dimensions, l_modes):
    """
    Reshape mode_matrix = np.zeros([m_samples*n_features, l_modes], dtype = np.float64)
    to mode_matrix_3D = np.zeros([m_samples, n_features, l_modes], dtype=np.float64)
    """
    n_features = dimensions[1]
    m_samples = dimensions[0]

    # note: POD_mode_matrix = np.zeros([m_samples*n_features, l_modes], dtype = np.float64)
    # note for POD modes, we choose to operate on only l_modes if l_modes is a digit, otheriwse l_modes = l_snapshots
    if l_modes in ('All', 'all'):
        # POD_mode_matrix transformed to 3-dimensional matrix initialized with 0.
        POD_mode_matrix_3D = np.zeros([m_samples, n_features, l_snapshots], dtype=np.float64)
    if isinstance(l_modes, int):
        # POD_mode_matrix transformed to 3-dimensional matrix initialized with 0.
        # l_modes + 1 because modes is calculated from 0
        POD_mode_matrix_3D = np.zeros([m_samples, n_features, l_modes + 1], dtype=np.float64)

    for idx_feature in range(n_features):
        start = m_samples * idx_feature
        end = m_samples * (idx_feature + 1)
        POD_mode_matrix_3D[:, idx_feature, :] = POD_mode_matrix[start:end, :]

    return POD_mode_matrix_3D


def get_corre_matrix(data_matrix, var_name: str = '', savepath: str = '', savename: str = "",
                     save: bool = False) -> np.ndarray:
    """
    Get the correlation matrix of shape [l_snapshots, l_snapshots].
    This is the reason snapshot pod takes much less time than normal POD
    """
    print("Building correlation matrix....")

    l_snapshots = data_matrix.shape[1]

    """same as below, but this one is slow
    corr_matrix = np.zeros([l_snapshots,l_snapshots],'float')
    start_time = clock.time()
    
    for t1 in range(0,l_snapshots,1) :
        for t2 in range(0,l_snapshots,1) :
            data1 = data_matrix[:,t1]
            data2 = data_matrix[:,t2]
            corr_matrix[t1,t2]=np.dot(data1,np.transpose(data2))

    corr_matrix = (1.0/l_snapshots) * corr_matrix
    print (corr_matrix)
    print ("Computed the correlation matrix in %.2f s" % (clock.time()-start_time))
    print ("Shape = {}, this square matrix has a shape of l*l\nAnd l is the number of snapshots =  {}.\n".format(corr_matrix.shape, l_snapshots))
    """
    ###
    start_time = clock.time()
    corr_matrix = (1.0 / l_snapshots) * np.dot(np.transpose(data_matrix), data_matrix)
    print("Computed the correlation matrix in %.2f s" % (clock.time() - start_time))
    ###

    if save:
        os.makedirs(savepath, exist_ok=True)
        np.savetxt(savepath + savename + '.txt', corr_matrix)

    return corr_matrix


def get_data_matrix(dimensions: np.ndarray = None, snapshots_dir: str = '', snapshot_time: np.ndarray = None,
                    var_name: str = '', savepath: str = '', savename: str = "data_matrix", save: bool = False):
    """
    Get data matrix from all snapshots, accumulate snapshots as column.
    Single snapshot shape [msamples, nfeatures] converted to [nfeatures*msamples, 1]

    return number of features and the data matrix of shape [nfeatures*msamples, l*snapshots].
    """

    m_samples = dimensions[0]
    n_features = dimensions[1]
    l_snapshots = dimensions[2]

    print("Getting data matrix....")

    start_time = clock.time()
    # Initialize the matrix
    
    matrix = np.zeros([m_samples * n_features, l_snapshots], dtype=np.float64);
    for idx_snapshots, time in enumerate(snapshot_time):
        path = snapshots_dir + str(time) + "/"
        new_path = update_path(path, var_name)

        df, mesh = import_vtk_data(path=new_path, var_name=var_name)
        matrix[:, idx_snapshots] = np.reshape(df.to_numpy(), -1, order = 'F') #np.hstack(df.to_numpy().T)

    print('Data matrix obtained in %.6f s.\n' % (clock.time() - start_time))
    """same as above loop, speed is roughly the same
    matrix = np.zeros([m_samples * n_features, l_snapshots], dtype=np.float64);

    for idx_snapshots, time in enumerate(snapshot_time):
        path = snapshots_dir + str(time) + "/"
        new_path = update_path(path, var_name)

        df, mesh = import_vtk_data(path=new_path, var_name=var_name)

        arr = np.zeros([m_samples * n_features], dtype=np.float64)
        print("Added snapshot = {}s to the data_matrix.".format(time))

        for idx_n_features in range(n_features):
            row = idx_n_features * m_samples
            column = (idx_n_features + 1) * m_samples
            arr[row:column] = df.iloc[:, idx_n_features]

        matrix[:, idx_snapshots] = arr

    print('Data matrix obtained in %.6f s.\n' % (clock.time() - start_time))
    """
    """same as above loop, speed is slower though
    start_time = clock.time()    
    test_data = []
    for idx_snapshots, time in enumerate(snapshot_time):
        path = snapshots_dir + str(time) + "/"
        new_path = update_path(path, var_name)
        
        df, mesh = import_vtk_data(path = new_path, var_name = var_name)

        test = df.unstack().reset_index()
        test_data.append(test.iloc[:,2].to_numpy())
    print ('Data matrix obtained in %.6f s.\n' % (clock.time()-start_time))
    """
    print("")

    if save:
        os.makedirs(savepath, exist_ok=True)
        np.savetxt(savepath + savename + '.txt', matrix)

    return matrix


def eigen_decomposition(corr_matrix, savepath: str = '', save: bool = False):
    """
    Get the eigenvalues and eigenvectors.
    """
    print("Performing eigen decompositions of data matrix....")

    start_time = clock.time()
    [value, vector] = np.linalg.eigh(corr_matrix)  # eig could give complex value, so we use eigh

    print('Eigenvalue decomposition executed in %.6f s.\n' % (clock.time() - start_time))

    value_arg = value.argsort(axis=0)[::-1]  # descending order argument

    value_ordered = np.sort(np.real(value), axis=0)[::-1]  # descending order value
    vector_ordered = vector[:, value_arg]  # sort column of vector based on value_arg

    if save:
        os.makedirs(savepath, exist_ok=True)
        np.savetxt(savepath + "eigen_value" + '.txt', value_ordered)
        np.savetxt(savepath + "eigen_vector" + '.txt', vector_ordered)

    return value_ordered, vector_ordered


def compute_modes(dimensions, eig_value, eig_vector, data_matrix, l_modes='0', savepath: str = '', save: bool = False):
    """
    Compute POD_modes_matrix of shape [m_samples, n_features, l_modes] where l_modes<l_snapshots
    as we don't need to compute all modes.
    """
    print("Computing modes....")

    eigen_value = eig_value
    eigen_vector = eig_vector

    l_snapshots = dimensions[2]

    if l_modes >= l_snapshots:
        print("Chosen number of modes is >= {}snapshots, exit!".format(l_snapshots))
        exit()

    # note: POD_mode_matrix = np.zeros([m_samples*n_features, l_modes], dtype = np.float64).
    # note for POD_mode_matrix, we choose to operate on only l_modes if l_modes is a digit,
    # otherwise l_modes = l_snapshots
    if l_modes in ('All', 'all'):
        print("Calculating {} modes.".format(l_modes))
        # after transpose, the row of time_coeff is the eigen_time
        time_coeff = np.transpose(np.sqrt(eigen_value * l_snapshots) * eigen_vector)
        # calculate spacial  POD_mode_matrix by mapping datamatrix on eigenvector, where the column of eigenvector is
        # the eigentime. Here, np.dot == np.matmul
        POD_mode_matrix = np.dot(data_matrix, eigen_vector) / np.sqrt(eigen_value * l_snapshots)

    if isinstance(l_modes, int):
        print("Calculating {} modes.".format(l_modes + 1))
        # after transpose, the row of time_coeff is the eigen_time
        time_coeff = np.transpose(np.sqrt(eigen_value[0:l_modes + 1] * l_snapshots) * eigen_vector[:, 0:l_modes + 1])
        # l_modes + 1 because modes is calculated from 0
        # Calculate spacial POD modes by mapping datamatrix on eigenvector, where the column of eigenvector
        # is the eigentime. Here, np.dot == np.matmul
        POD_mode_matrix = np.dot(data_matrix, eigen_vector[:, 0:l_modes + 1]) / np.sqrt(
            eigen_value[0:l_modes + 1] * l_snapshots)  # snapshot is calculated from 1

    if save:
        np.savetxt(savepath + "time_coeff" + "_{}mode.txt".format(l_modes + 1), time_coeff)
        np.savetxt(savepath + "POD_{}mode_matrix.txt".format(l_modes + 1), POD_mode_matrix)

    print("Saved time coeff and POD mode matrix for {} modes.\n".format(l_modes + 1))

    return time_coeff, POD_mode_matrix


def reconstruct_modes(dimensions, time_coeff, POD_mode_matrix, n_snapshots_to_reconstruct=1, n_modes_to_reconstruct=0, var_name: str = '', savepath: str = ''):
    """
    Compute from "POD_mode_matrix" of shape = [n_features*m_samples, l_snapshots] or [n_features*m_samples, l_modes]
    and from "time_coeff" of shape = [l_snapshots, l_snapshots] or [l_modes, l_modes],

    reconstructed_mode_matrix = np.zeros([m_samples*n_features, l_snapshots, l_modes+1], dtype = np.float64)
    """
    savepath = savepath + '../' + var_name + 'reconstructedmodes/'
    os.makedirs(savepath, exist_ok=True)
    time_coeff = time_coeff

    m_samples = dimensions[0]
    n_features = dimensions[1]
    l_snapshots = dimensions[2]

    if n_snapshots_to_reconstruct > l_snapshots:
        print ("No sufficient snapshots available to reconstruct.")
        exit()

    if n_modes_to_reconstruct in ('All', 'all'):
        reconstructed_mode = np.dot(POD_mode_matrix, time_coeff)
        # POD_mode transformed to 3-dimensional matrix initialized with 0.
        # This doesn't seem to be working!!!!!!!!!!!
        reconstructed_mode_matrix = np.zeros([m_samples, n_features, l_snapshots], dtype=np.float64)

    if isinstance(n_modes_to_reconstruct, int):
        reconstructed_mode_matrix = np.zeros([m_samples * n_features, n_snapshots_to_reconstruct, n_modes_to_reconstruct + 1], dtype=np.float64)
        for mode in np.arange(n_modes_to_reconstruct + 1):
            # 0 column of POD_mode_matrix is the actual mode for mode 0: phi
            # 0 row of time_coeff is the time coefficient of mode 0: a(t)
            # we reconstruct mode 0 as a(t0) * phi + a(t1) * phi + ....
            # column of reconstructed mode corresponds to the mode variation in time.
            #print ("mode: ", mode)
            #print (" POD_mode_matrix Shape = ", POD_mode_matrix[:, mode].shape)
            #print ("time_coeff Shape = ", time_coeff.shape)
            reconstructed_mode = np.outer(POD_mode_matrix[:, mode], time_coeff[mode, :n_snapshots_to_reconstruct])
            reconstructed_mode_matrix[:, :n_snapshots_to_reconstruct, mode] = reconstructed_mode

    return reconstructed_mode_matrix


def export_reconstructed_mode_vtk(dimensions, reconstructed_mode_matrix, mesh, n_snapshots_to_export=1, n_modes_to_export=0,
                                  var_name: str = '', savename: str = "", savepath: str = ''):
    """
    Export vtk containing appended l_modes where l_modes < l_snapshots
    reconstructed_mode_matrix = np.zeros([m_samples*n_features, l_snapshots, l_modes+1], dtype = np.float64);
    
    if n_modes_to_export = 3, we export 0, 1, 2 individually.
    if n_modes_to_export = [0, 1, 2], we accumulate modes 0, 1, and 2
    """

    savepath = savepath + '../' + var_name + 'reconstructedmodes/'
    os.makedirs(savepath, exist_ok=True)
    mesh.clear_data()

    n_features = dimensions[1]
    m_samples = dimensions[0]
    l_snapshots = dimensions[2]

    if n_snapshots_to_export > l_snapshots:
        print("Insufficient snapshots to be exported, max n_snapshots_to_export = {}".format(l_snapshots))
        exit()

    if isinstance(n_modes_to_export, list):
        print("Detected listed modes, export accumulated modes!")
        mode_2D = reconstructed_mode_matrix[:, :, n_modes_to_export]
        #print(mode_2D.shape)
        mode_2D = np.sum(mode_2D, axis=2)
        #print(mode_2D.shape)
        for idx_snapshots in range(n_snapshots_to_export):
            print("    Exporting snapshot {}.".format(idx_snapshots))
            path = savepath + "{}{}.".format(savename, n_modes_to_export) + str(idx_snapshots) + ".vtk"
            mode_1D = mode_2D[:, idx_snapshots]
            #print(mode_1D.shape)
            for idx_features in range(n_features):
                start = m_samples * idx_features
                end = m_samples * (idx_features + 1)
                mesh[var_name + ':' + str(idx_features)] = mode_1D[start:end]
                mesh.save(path)

    elif isinstance(n_modes_to_export, int):
        print("Detected integer modes, export individual modes!")
        for idx_mode in np.arange(n_modes_to_export):
            print("Dealing with Mode {}.".format(idx_mode))
            mode_2D = reconstructed_mode_matrix[:, :, idx_mode]  # 2D shape [m_samples*n_features, l_snapshots]

            for idx_snapshots in range(n_snapshots_to_export):
                print("    Exporting snapshot {}.".format(idx_snapshots))
                path = savepath + "{}{}.".format(savename, idx_mode) + str(idx_snapshots) + ".vtk"
                mode_1D = mode_2D[:, idx_snapshots]

                for idx_features in range(n_features):
                    start = m_samples * idx_features
                    end = m_samples * (idx_features + 1)
                    mesh[var_name + ':' + str(idx_features)] = mode_1D[start:end]
                    mesh.save(path)


def export_mode_vtk(mode, mesh, var_name: str = '', savename: str = "", savepath: str = ''):
    """
    Export vtk contaning appended l_modes where l_modes < l_snapshots
    """
    print("Exporting mode vtk....")

    savepath = savepath + '../' + var_name + 'modes/'
    os.makedirs(savepath, exist_ok=True)
    mesh.clear_data()

    n_features = mode.shape[1]
    m_samples = int(mode.shape[0] / n_features)
    l_snapshots = mode.shape[2]

    for idx_snapshots in range(l_snapshots):
        path = savepath + "{}".format(savename) + str(idx_snapshots) + ".vtk"
        print("Dealing with Mode {}.".format(idx_snapshots))
        for idx_features in range(n_features):
            start = m_samples * idx_features
            end = m_samples * (idx_features + 1)

            print("    Add {}:{} into vtk mesh.".format(var_name, idx_features))
            mesh[var_name + ':' + str(idx_features)] = mode[:, idx_features, idx_snapshots]

        mesh.save(path)
        print("{} {} written to {}.\n".format(savename, idx_snapshots, path))


def plt_eigen_value_spectrum(eig_value: np.array = None, modeminmax=[float, float],
                             save: bool = False, figname: str = None, figpath: str = None):
    """
    Plot eigen_value_spectrum using sorted eigenvalues
    modeminmax: e.g., [1, 50], plot shows turbulent kinetic energy of modes ranging from 1 to 50.
    """

    x = np.arange(modeminmax[0], modeminmax[1])
    y = eig_value[modeminmax[0]: modeminmax[1]] / sum(eig_value[1:]) * 100
    #print(eig_value, sum(eig_value))
    fig, ax = _set_plot_settings()
    ax.scatter(x, y, marker="s", color="black")
    # ax.plot(x, y, color="black", ls="-")
    ax.set_xlabel('POD Mode Number', fontsize=16)
    ax.set_ylabel('Normalized TKE(%) in Modes \n [{} - {}] '.format(modeminmax[0], modeminmax[1]), fontsize=16)
    ax.set_ylim(-0.1, 1.1 * np.max(y))

    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(y), color="red", ls="--")
    ax2.set_ylabel('Accumulation of \n Normalized TKE (%)', fontsize=16)
    ax2.set_ylim(-0.1, 100)

    fig.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def plt_power_spectrum(time_coeff: np.array = None, snapshot_time: np.array = None, Strouhal: bool = False, l_modes='0',
                       save: bool = False, figname: str = None, figpath: str = None):
    """
    Plot power spectrum of time_coeff  (normalised amplitude vs frequency)
    if time_coeff has only say 3 rows, they corresponds to the output of 3 modes used in 'compute_modes'.
    the l_modes should not be larger than the number of rows in time_coeff.txt
    """
    from scipy import signal

    l_modes_avail = len(time_coeff)
    fig, ax = _set_plot_settings()
    linestyle = list(lines.lineStyles.keys())
    f = 1 / (snapshot_time[1] - snapshot_time[0])

    if Strouhal:
        print ("To calculate Strouhal number, you need to, ")
        xlabel = 'Strouhal Number'
        D = float(input("Input the Characteristic length [m] (Enter to Confirm):"))
        U = float(input("Input the Characteristic velocity [m/s] (Enter to Confirm):"))
    else:
        xlabel = 'Frequency [Hz]'
        D = 1
        U = 1

    if l_modes in ('All', 'all'):
        for row in np.arange(l_modes_avail):
            frequency, PSD = signal.welch(time_coeff[row, :], window='hanning', fs=f, detrend='constant')
            PSD_max = max(abs(PSD))
            # ax.plot(snapshot_time, time_coeff[row,:], label = "Mode {}".format(row), ls = linestyle[row])
            ax.plot(frequency * D / U, PSD / PSD_max, label="Mode {}".format(row), ls=linestyle[row])
    elif l_modes_avail < int(l_modes):
        print(
            "Time_coeff for the requested mode is not saved, please re-run function 'compute_modes' using 'l_modes'> {}".format(
                l_modes))
        exit()
    else:
        frequency, PSD = signal.welch(time_coeff[int(l_modes), :], window='hanning', fs=f, detrend='constant')
        PSD_max = max(abs(PSD))
        ax.plot(frequency * D / U, PSD / PSD_max, label="Mode {}".format(int(l_modes)), ls=linestyle[int(l_modes)])

    ax.legend(handlelength=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel=xlabel, fontsize=16)
    ax.set_ylabel('Power Spectrum \n (normalised)', fontsize=16)

    fig.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def plt_phase_portrait(time_coeff: np.array = None, modes=None, save: bool = False,
                       figname: str = None, figpath: str = None):
    if modes is None:
        modes = [1, 2]
    figname = figname + str(modes[0]) + "-" + str(modes[1])
    fig, ax = _set_plot_settings()
    ax.scatter(time_coeff[modes[0], :], time_coeff[modes[1], :], color="blue")
    ax.set_aspect('equal', 'datalim')

    ax.set_xlabel(xlabel="a{}(t)".format(modes[0]), fontsize=16)
    ax.set_ylabel(ylabel="a{}(t)".format(modes[1]), fontsize=16)

    fig.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)

def _set_plot_settings(nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, plt.Subplot]:
    fig, ax = plt.subplots(nrows, ncols, figsize=[7.2, 4])
    # plt.gca().set_aspect('equal', 'datalim')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig, ax


def _save_fig(save: bool = False, figname: str = None, figpath: str = None):
    if save:
        plt.savefig(figpath + figname + '.png')
    else:
        plt.show()


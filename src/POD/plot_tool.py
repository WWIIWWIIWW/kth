import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
from typing import Union, Sequence, Tuple, Type

plt.rc('xtick', labelsize='16')
plt.rc('ytick', labelsize='16')
plt.rc('font', family='serif')
plt.rc('text', usetex=False)
plt.rc('legend', fontsize=13)
plt.rcParams.update({'figure.max_open_warning': 0})


def plt_eigen_value_spectrum(eig_value: np.array = None, modeminmax=[float, float],
                             save: bool = False, figname: str = None, figpath: str = None) -> None:
    """
    Plot eigen_value_spectrum using sorted eigenvalues
    modeminmax: e.g., [1, 50], plot shows turbulent kinetic energy of modes ranging from 1 to 50.
    """

    x = np.arange(modeminmax[0], modeminmax[1])
    y = eig_value[modeminmax[0]: modeminmax[1]] / sum(eig_value[modeminmax[0]:]) * 100

    fig, ax = _get_plot_settings()
    ax.scatter(x, y, marker="s", color="black", label='Mode TKE')

    # ax.plot(x, y, color="black", ls="-")
    ax.set_xlabel('POD Mode Number', fontsize=16)
    ax.set_ylabel('Normalized TKE(%) in Modes \n [{} - {}] '.format(modeminmax[0], modeminmax[1]), fontsize=16)
    ax.set_ylim(0, np.ceil(1.1 * np.max(y)))

    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(y), color="red", ls="--", label='Accumu. TKE')
    ax2.set_ylabel('Accumulation of \n Normalized TKE (%)', fontsize=16)
    ax2.set_ylim(-0.1, 100)

    # ax.legend(loc='best')
    ax2.legend(loc='upper right')

    fig.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def plt_power_spectrum(time_coeff: np.array = None, snapshot_time: np.array = None, Strouhal: bool = False, l_modes='0',
                       save: bool = False, figname: str = None, figpath: str = None) -> None:
    """
    Plot power spectrum of time_coeff  (normalised amplitude vs frequency)
    if time_coeff has only say 3 rows, they corresponds to the output of 3 modes used in 'compute_modes'.
    the l_modes should not be larger than the number of rows in time_coeff.txt
    """
    from scipy import signal

    l_modes_avail = len(time_coeff)
    fig, ax = _get_plot_settings()
    linestyle = list(lines.lineStyles.keys())
    f = 1 / (snapshot_time[1] - snapshot_time[0])

    if Strouhal:
        print("To calculate Strouhal number, you need to, ")
        xlabel = 'Strouhal Number'
        D = float(input("Input the Characteristic length [m] (Enter to Confirm):"))
        U = float(input("Input the Characteristic velocity [m/s] (Enter to Confirm):"))
    else:
        xlabel = 'Frequency [Hz]'
        D = 1
        U = 1

    if l_modes in ('All', 'all'):
        for row in np.arange(l_modes_avail):
            print("?")
            frequency, PSD = signal.welch(time_coeff[row, :], window='hanning', fs=f, detrend='constant')
            print(frequency, PSD)
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


def plt_phase_portrait(time_coeff: np.array = None, modes: list = None, JPDF: bool = True, save: bool = False,
                       figname: str = None, figpath: str = None) -> None:
    if modes is None:
        modes = [1, 2]
    figname = figname + str(modes[0]) + "-" + str(modes[1])
    fig, ax = _get_plot_settings()

    if JPDF == True:
        import seaborn as sns
        sns.set_style('darkgrid')
        ax = sns.jointplot(time_coeff[modes[0], :], time_coeff[modes[1], :], cmap=cm.jet,
                           kind='kde', fill=False)  # 'hot_r'
        ax.ax_joint.set_xlabel(xlabel="a{}(t)".format(modes[0]), fontsize=16)
        ax.ax_joint.set_ylabel(ylabel="a{}(t)".format(modes[1]), fontsize=16)
    else:
        ax.scatter(time_coeff[modes[0], :], time_coeff[modes[1], :], color="blue")
        ax.set_aspect('equal', 'datalim')

        ax.set_xlabel(xlabel="a{}(t)".format(modes[0]), fontsize=16)
        ax.set_ylabel(ylabel="a{}(t)".format(modes[1]), fontsize=16)

    plt.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def plt_spatiotemporal(data_matrix, snapshot_time: np.array = None,
                       coord: np.array = None, boolDict: dict = [],
                       save: bool = False, figname: str = None,
                       figpath: str = None) -> None:
    """
    Make sure the component you choose is the same as the lineVector used in get_rcs().
    This should be done automatically, but not implemented yet.

    Note for wall stress plot, we need to multiply dataMatrix by -1 to make sure stress and U gradient has same sign.
    """

    if data_matrix.shape[0] > len(boolDict):
        component = int(input('Type 1, 2, 3 to choose xyz component of vector (Enter to Confirm):'))
        start = (component - 1) * len(boolDict)
        end = component * len(boolDict)
        data_matrix = data_matrix[start:end, :]

    x_, y_ = np.meshgrid(snapshot_time, coord)
    # print (x_.shape, y_.shape,boolDict.shape,  data_matrix.shape, data_matrix[boolDict].shape)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7.2, 7.2])

    im = ax.contourf(y_, x_, data_matrix[boolDict], cmap=cm.jet)

    xlabel, ylabel, cbarlabel = _get_label(cbar=True)

    ax.set_xlabel(xlabel=xlabel, fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)

    if input('change xy_lim? (True/False):') == 'True':
        xim, ylim = _get_limit()
        ax.set_xlim(xim)
        ax.set_ylim(ylim)

    """
    if input('change cbar_lim? (True/False):') == 'True':
        vmin = float(input("Input Cbar Min. for your plot (Enter to Confirm):"))
        vmax = float(input("Input Cbar Max. for your plot (Enter to Confirm):"))
        im.set_clim(vmin, vmax)
    """

    cbar = fig.colorbar(im, orientation='vertical')
    cbar.ax.set_ylabel(cbarlabel, size=15)

    plt.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def plt_PDF(data_matrix, location: list = [],
            xlim: list = [],
            coord: np.array = None, boolDict: dict = [],
            save: bool = False, figname: str = None,
            figpath: str = None) -> None:

    fig, ax = _get_plot_settings()
    data = data_matrix[boolDict]  # row = Nu at coord, column = snapshot
    data_max = np.amax(data)
    color = ['black', 'red', 'cyan', 'green', 'orange', 'purple']
    for idx, value in enumerate(location):
        row_idx = find_nearest(coord, float(value))
        ax.hist(data[row_idx]/data_max, bins=50, density=True, rwidth=0.5, color=color[idx], label = 'median = {:.2f}, std = {:.3f}'.format(float(np.median(data[row_idx])/data_max), np.std(data[row_idx]/data_max)))
    ax.set_xlim(xlim)
    ax.legend()
    plt.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)

def plt_PDF_negSS(data_matrix,
                 coord: np.array = None, boolDict: dict = [],
                 save: bool = False, figname: str = None,
                 figpath: str = None) -> None:
    """
    plot probability of negative wall shear stress along radial direction (probability collected over time)
    """
    #this may be useless now since we choose data from get_data_matrix by specifying case_as_scalar = True
    if data_matrix.shape[0] > len(boolDict):
        component = int(input('Type 1, 2, 3 to choose xyz component of vector (Enter to Confirm):'))
        start = (component - 1) * len(boolDict)
        end = component * len(boolDict)
        data_matrix = data_matrix[start:end, :]

    fig, ax = _get_plot_settings()
    arr_2d = data_matrix[boolDict]  # row = SS at coord, column = snapshot

    # Value to compare to
    compare_val = 0

    # Get density of values smaller than compare_val along each row
    density = np.sum(arr_2d <= compare_val, axis=1) / arr_2d.shape[1]
    np.savetxt(figname + ".txt", density)
    im = ax.plot(coord, density, ls="-")

    xlabel, ylabel = _get_label(cbar=False)
    xim, ylim = _get_limit()

    ax.set_xlabel(xlabel=xlabel, fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)
    ax.set_xlim(xim)
    ax.set_ylim(ylim)

    plt.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)

def plt_JPDF(data_matrix, snapshot_time: np.array = None,
             coord: np.array = None, boolDict: dict = [],
             save: bool = False, figname: str = None,
             figpath: str = None) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7.2, 7.2])
    data = data_matrix[boolDict]  # row = Nu at coord, column = snapshot
    idx = find_nearest(coord, 0.005)
    idx_2 = find_nearest(coord, 0.01)

    ax.hist(data[idx], bins=50, density=True, rwidth=0.5, color='skyblue')
    ax.hist(data[idx_2], bins=50, density=True, rwidth=0.5, color='black')
    ax.legend()
    plt.show()


def get_rcs(mesh: list = None, lineVector: list = [0, 1, 0],
            lineNormalVector: list = [1, 0, 0]):
    """
    Get idx = index or BoolDict = True/False
    to sort data, return also coordinate
    along a radial direction.

    """
    points = mesh.points
    pos_idx = lineVector.index(1)  # along which we define the line
    pos_idx2 = [x + y for x, y in zip(lineVector, lineNormalVector)].index(0)  # along which we find coordinate

    minValue = np.unique(abs(points[:, pos_idx2])).min()  # was idx, change to idx2
    boolDict = (points[:, pos_idx2] == minValue) & (points[:, pos_idx] >= 0)
    idx = [i for i, val in enumerate(boolDict) if val]  # index of points

    coord = points[boolDict][:, pos_idx]  # we find coord along y direction, was idx2, now idx

    return boolDict, coord


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_index_pdf(coord):
    unique = np.unique(coord)
    for val in unique:
        idx_array = np.where(np.isclose(coord, val, 1e-4))
    return idx_array


"""
         if len(idx_array[0]) > 5:
             print ("value =", val)
             print ("matrix = ", coord[idx_array])
"""


# correltion =


def get_mutual_index(matrix1, matrix2):
    from sklearn.feature_selection import mutual_info_regression
    score = mutual_info_regression(matrix1, matrix2)
    return score


def plt_MeanvsRadial(mean, coord: np.array = None,
                     save: bool = False, figname: str = None,
                     figpath: str = None) -> None:
    y_ = mean  # to take the mean of each row
    x_ = coord

    fig, ax = _get_plot_settings()
    im = ax.plot(x_, y_, ls="--")

    xlabel, ylabel = _get_label(cbar=False)
    xim, ylim = _get_limit()

    ax.set_xlabel(xlabel=xlabel, fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)
    ax.set_xlim(xim)
    ax.set_ylim(ylim)

    plt.tight_layout()

    _save_fig(save, figname, figpath)

    plt.close(fig)


def _get_plot_settings(nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, plt.Subplot]:
    fig, ax = plt.subplots(nrows, ncols, figsize=[7.2, 4])
    # plt.gca().set_aspect('equal', 'datalim')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig, ax


def _save_fig(save: bool = False, figname: str = None, figpath: str = None) -> None:
    if save:
        plt.savefig(figpath + figname + '.png')
    else:
        plt.show()


def _get_label(cbar: bool = False) -> None:
    xlabel = str(input("Input string for the xlabel (Enter to Confirm):"))
    ylabel = str(input("Input string for the ylabel (Enter to Confirm):"))
    if cbar:
        cbarlabel = str(input("Input string for the cbarlabel (Enter to Confirm):"))
        return xlabel, ylabel, cbarlabel
    else:
        return xlabel, ylabel


def _get_limit() -> None:
    xmin = float(input("Input Min. X for your plot (Enter to Confirm):"))
    xmax = float(input("Input Max. X for your plot (Enter to Confirm):"))
    ymin = float(input("Input Min. Y for your plot (Enter to Confirm):"))
    ymax = float(input("Input Max. Y for your plot (Enter to Confirm):"))

    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    return xlim, ylim

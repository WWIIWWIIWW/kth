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
                             save: bool = False, figname: str = None, figpath: str = None):
    """
    Plot eigen_value_spectrum using sorted eigenvalues
    modeminmax: e.g., [1, 50], plot shows turbulent kinetic energy of modes ranging from 1 to 50.
    """

    x = np.arange(modeminmax[0], modeminmax[1])
    y = eig_value[modeminmax[0]: modeminmax[1]] / sum(eig_value[1:]) * 100
    # print(eig_value, sum(eig_value))
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


def plt_phase_portrait(time_coeff: np.array = None, modes: list = None, JPDF: bool = True, save: bool = False,
                       figname: str = None, figpath: str = None):
    if modes is None:
        modes = [1, 2]
    figname = figname + str(modes[0]) + "-" + str(modes[1])
    fig, ax = _set_plot_settings()

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


def plt_spatiotemporal(data_matrix, snapshot_time: np.array = None, mesh: list = None, lineVector: list = [0, 1, 0],
                       lineNormalVector: list = [1, 0, 0], save: bool = False, figname: str = None,
                       figpath: str = None) -> None:

    points = mesh.points
    pos_idx = lineVector.index(1)  # along which we define the line
    pos_idx2 = [x + y for x, y in zip(lineVector, lineNormalVector)].index(0)  # along which we find long coordinate
    minValue = np.unique(abs(points[:, pos_idx])).min()
    boolDict = (points[:, pos_idx] == minValue) & (points[:, pos_idx2] >= 0)
    x_, y_ = np.meshgrid(snapshot_time, points[boolDict][:, pos_idx2])

    fig, ax = _set_plot_settings()
    im = ax.contourf(y_, x_, data_matrix[boolDict], cmap=cm.jet)
    xlabel, ylabel, cbarlabel = _set_label()
    ax.set_xlabel(xlabel=xlabel, fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)
    cbar = fig.colorbar(im, orientation='vertical')
    cbar.ax.set_ylabel(cbarlabel, size=15)
    plt.tight_layout()

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


def _set_label():
    xlabel = str(input("Input string for the xlabel (Enter to Confirm):"))
    ylabel = str(input("Input string for the ylabel (Enter to Confirm):"))
    cbarlabel = str(input("Input string for the cbarlabel (Enter to Confirm):"))
    return xlabel, ylabel, cbarlabel
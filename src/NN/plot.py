import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, Type, Dict

key = Union[str, str]

def show(history):
    print(history.history.keys())
    
def _set_plot_settings(dim: int = 2, figsize: tuple = (7.2, 7.2)) -> Tuple[plt.Figure, plt.Subplot]:
    fig = plt.figure(figsize = figsize)
    if dim == 2:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
    elif dim == 3:
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_zlabel('', fontsize = 16, labelpad=-10)
        ax.view_init(-140, 60) #(elevation angle, azimuth angle)
    ax.set_xlabel('', fontsize = 16, labelpad=-10)
    ax.set_ylabel('', fontsize = 16, labelpad=-10)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    return fig, ax

def _save_fig(save: bool = False, figname: str = None, figpath: str = None):
    plt.tight_layout()
    if figpath:
        os.makedirs(figpath, exist_ok = True)

    if save:
        plt.savefig(figpath+figname+'.png', bbox_inches='tight', dpi=180)
    else:
        plt.show()
    plt.close('all')
    
def plot(history, xy: Sequence[key] = list(), save: bool = False, figname: str = None, figpath: str = None):

    dim = len(xy)
    fig, ax = _set_plot_settings(dim)
    
    his_key = xy[1]

    ax.set_aspect("auto") #reset to auto
    ax.plot(history.history[his_key])

    ax.set_title('model ' + his_key)
    ax.set_xlabel(xy[0])
    ax.set_ylabel(xy[1])
    ax.legend(['train'], loc='upper left')

    if xy.count(his_key) == 2:
        ax.plot(history.history['val_'+his_key])
        ax.legend(['train', 'test'], loc='upper left')
    
    _save_fig(save, figname, figpath)

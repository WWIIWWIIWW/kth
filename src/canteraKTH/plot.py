import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, Type

def _set_plot_settings(row = 1, col = 1) -> Tuple[plt.Figure, plt.Subplot]:
    plt.rc('xtick', labelsize='16')
    plt.rc('ytick', labelsize='16')
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    plt.rc('legend', fontsize = 11)
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    fig_row_length = row * 4
    fig_col_length = col * 7.2

    if col == 1:
        print ("Share x axis!")
        fig, ax = plt.subplots(row, col, sharex=True, sharey=False, figsize = [fig_col_length + col * 0.4, fig_row_length])
    elif row == 1:
        print ("Share y axis!")
        fig, ax = plt.subplots(row, col, sharex=False, sharey=True, figsize = [fig_col_length, fig_row_length + row * 0.4])
        
    fig.add_subplot(111, frameon=False)
    #plt.gca().set_aspect('equal', 'datalim')
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    
    return fig, ax
    


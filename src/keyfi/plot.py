import warnings
import umap.plot
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.patches import Patch
from matplotlib.patches import Circle

from hdbscan import HDBSCAN, all_points_membership_vectors
from matplotlib import colors
from matplotlib import rcParams
from matplotlib.patches import Patch
from typing import Union, Sequence, Tuple, Type, Dict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


Num = Union[int, float]
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

def fix_yticks(labels):
    new_yticks = []
    for label in labels:
        if '_' in label.get_text():
            # Fix species' names
            lst = label.get_text().split('_')
            lst.insert(1, '_{')
            lst.append('}')
            new_yticks.append(''.join(lst))
        elif ':' in label.get_text():
            # Fix velocity componenets' names
            lst = label.get_text().split(':')
            if lst[-1] == '0':
                lst[-1] = '_x'
            elif lst[-1] == '1':
                lst[-1] = '_y'
            elif lst[-1] == '2':
                lst[-1] = '_z'
            new_yticks.append(''.join(lst))
        else:
            new_yticks.append(label.get_text())

    new_yticks = ['$\mathrm{' + item + '}$' for item in new_yticks]
    return new_yticks


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
    
     
def _set_colors(n_clusters: int) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    cmap = colors.ListedColormap(tuple(sns.color_palette('bright', n_clusters)))
    norm = colors.BoundaryNorm(np.arange(-0.5, n_clusters), n_clusters)
    return cmap, norm


def _set_colorbar(im, fig, ax, label: str = None, dim: int = 2):
    if dim == 3:
        kwargs = {'width': "100%", 'height': "5%", 'loc': 'lower center', }
        axins = inset_axes(ax, borderpad=-3, **kwargs)
        cb = fig.colorbar(im, cax = axins, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
    elif dim == 2:
        kwargs = {'width': "5%", 'height': "100%", 'loc': 'center right'}
        axins = inset_axes(ax, borderpad=-3, **kwargs)
        cb = fig.colorbar(im, cax = axins, orientation='vertical')
    
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label, size=16)
    
    
def _set_legend(labels: np.ndarray, cmap: colors.ListedColormap, ax: plt.Subplot, dim: int = 2):
    unique_labels = np.unique(labels)
    legend_elements = [Patch(facecolor=cmap.colors[i], label=unique_label)
                       for i, unique_label in enumerate(unique_labels)]
    legend = ax.legend(handles=legend_elements, title='Clusters', fontsize=14,
                       title_fontsize=14, loc="center left")
    if dim == 3:
        ax.set_legend(loc = "center left")          
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.25))


def _remove_axes(ax: plt.Subplot, dim: int = 2):
    ax.set(yticklabels=[], xticklabels=[])
    if dim == 3:
        ax.set(yticklabels=[], xticklabels=[], zticklabels=[]) 
    #ax.tick_params(left=False, bottom=False)


def _set_point_size(points: np.ndarray) -> np.ndarray:
    point_size = 100.0 / np.sqrt(points.shape[0])
    return point_size


def _set_cluster_member_colors(clusterer: HDBSCAN, soft: bool = True):
    n_clusters = np.size(np.unique(clusterer.labels_))

    if -1 in np.unique(clusterer.labels_) and not soft:
        color_palette = sns.color_palette('bright', n_clusters-1)
    else:
        color_palette = sns.color_palette('bright', n_clusters)

    if soft:
        soft_clusters = all_points_membership_vectors(clusterer)
        cluster_colors = [color_palette[np.argmax(x)]
                          for x in soft_clusters]
    else:
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p)
                             for x, p
                             in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_member_colors, color_palette


def _save_fig(save: bool = False, figname: str = None, figpath: str = None):

    if figpath:
        os.makedirs(figpath, exist_ok = True)

    if save:
        plt.savefig(figpath+figname+'.png', bbox_inches='tight', dpi=180)
    else:
        plt.show()
    plt.close('all')

def plot_embedding(embedding: np.ndarray, data: pd.DataFrame = pd.DataFrame(), scale_points: bool = True, label: str = None, cmap_var: str = None, cmap_minmax: Sequence[Num] = list(), save: bool = False, figname: str = None, figpath: str = None):
    """
    embedding ----
                 |
                 ---->  t-SNE map
                 |
    dataFrame ----
    """
    
    if cmap_var not in data.columns and cmap_var:
        raise ValueError(
            'invalid variable for the color map. Expected one of: %s' % data.columns)

    if len(cmap_minmax) != 2 and cmap_minmax:
        raise ValueError(
            'too many values to unpack. Expected 2')
    
    if not label:
        label = cmap_var 

    dim = embedding.shape[1]  
    fig, ax = _set_plot_settings(dim)
        
    if scale_points:
        point_size = _set_point_size(embedding)
    else:
        point_size = None
                
    if cmap_var:
        if cmap_minmax:
            im = ax.scatter(*embedding.T, s=point_size, c=data[cmap_var],  vmin=cmap_minmax[0], vmax=cmap_minmax[1], cmap='rainbow')
        else:
            im = ax.scatter(*embedding.T, s=point_size, c=data[cmap_var], cmap='rainbow')
        _set_colorbar(im, fig, ax, label=label, dim=dim)
    else:
        ax.scatter(*embedding.T, s=point_size)

    _remove_axes(ax, dim=dim)
    _save_fig(save, figname, figpath)

def plot_manifold(cluster_labels: np.ndarray, data: pd.DataFrame = pd.DataFrame(), xvar: str = None, yvar: str = None, save: bool = False, figname: str = None, figpath: str = None):
    """
    dataFrame       ----
                       |
                       ---->  manifold map
                       |
    cluster_labels  ----
    """
    #if xvar or yvar not in data.columns:
    #    raise ValueError(
    #        'invalid feature for the manifold map. Expected one of: %s' % data.columns)
            
    n_clusters = np.size(np.unique(cluster_labels))
    cmap, norm = _set_colors(n_clusters)

    fig, ax = _set_plot_settings()

    ax.set_aspect("auto") #reset to auto
    
    ax.scatter(data[xvar], data[yvar], s=1,
                c=cluster_labels, cmap=cmap, norm=norm)

    if n_clusters <= 12:
        _set_legend(labels=cluster_labels, cmap=cmap, ax=ax)
    else:
        _set_colorbar(label='Clusters', ticks=np.arange(n_clusters))
    
    if xvar == 'Z':
        _add_fl(ax)
        
    ax.set_xlim([data[xvar].min(), data[xvar].max()])
    ax.set_ylim([data[yvar].min(), data[yvar].max()])
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    
    if yvar == 'CEMA_texplog':
        ax.set_ylabel('CM')
    
    _save_fig(save, yvar+figname, figpath)

def _add_fl(ax: plt.Subplot):
    ax.axvline(x=0.027, c ='b', ls = '--')
    ax.axvline(x=0.090, c ='b', ls = '--')
    ax.axvline(x=0.055, c ='r', ls = '--')
    
    locs = ax.yaxis.get_majorticklocs()
    text_loc = (locs[0]-locs[1])/8
    print (ax.get_yticks())
    text_loc = ax.get_yticks()[-3]
    ax.text(x=0.027, y = text_loc, s = '$Z_{fl,l}$', fontsize = 14)
    ax.text(x=0.090, y = text_loc, s = '$Z_{fl,u}$', fontsize = 14)
    ax.text(x=0.055, y = text_loc, s = '$Z_{st}$', fontsize = 14)
    
def plot_clustering(embedding: np.ndarray, cluster_labels: np.ndarray, scale_points: bool = True, save: bool = False, figname: str = None, figpath: str = None):
    """
    embedding       ----
                       |
                       ---->    cluster map
                       |
    cluster_labels  ----
    """
    #fig, ax = _set_plot_settings() ->check if this should be removed
    n_clusters = np.size(np.unique(cluster_labels))

    if n_clusters > 30:
        warnings.warn(
            'Number of clusters (%s) too large, clustering visualization will be poor' % n_clusters)

    cmap, norm = _set_colors(n_clusters)

    dim = embedding.shape[1]  
    fig, ax = _set_plot_settings(dim)
    
    if scale_points:
        point_size = _set_point_size(embedding)
    else:
        point_size = None

    ax.scatter(*embedding.T, s=point_size,
                c=cluster_labels, cmap=cmap, norm=norm)

    if n_clusters <= 12:
        _set_legend(labels=cluster_labels, cmap=cmap, ax=ax)
    else:
        _set_colorbar(label='Clusters', ticks=np.arange(n_clusters))
    
    _remove_axes(ax, dim)
    _save_fig(save, figname, figpath)


def plot_cluster_membership(embedding: np.ndarray, clusterer: HDBSCAN, scale_points: bool = True, legend: bool = True, save: bool = False, figname: str = None, figpath: str = None, soft: bool = True):

    n_clusters = np.size(np.unique(clusterer.labels_))

    if n_clusters > 12:
        warnings.warn(
            'Number of clusters (%s) too large, clustering visualization will be poor' % n_clusters)

    cluster_member_colors, color_palette = _set_cluster_member_colors(
        clusterer, soft)
        
    dim = embedding.shape[1]  
    fig, ax = _set_plot_settings(dim)
        
    if scale_points:
        point_size = 5*_set_point_size(embedding)
    else:
        point_size = 20
    
    ax.scatter(*embedding.T, s=point_size, linewidth=0,
                c=cluster_member_colors, alpha=0.5)

    if legend:
        if -1 in np.unique(clusterer.labels_):
            unique_colors = ((0.5, 0.5, 0.5), *tuple(color_palette))
        else:
            unique_colors = tuple(color_palette)
        cmap = colors.ListedColormap(unique_colors)
        _set_legend(labels=clusterer.labels_, cmap=cmap, ax=ax)

    _remove_axes(ax, dim)
    _save_fig(save, figname, figpath)


def umap_plot(mapper: Type, save: bool = False, figname: str = None, figpath: str = None, **kwargs):
    umap.plot.points(mapper, **kwargs)
    
    _save_fig(save, figname, figpath)
    
    
def new_set_cluster_member_colors(clusterer: HDBSCAN, soft: bool = True):
    n_clusters = np.size(np.unique(clusterer.labels_))

    if -1 in np.unique(clusterer.labels_) and not soft:
        color_palette = sns.color_palette('bright', n_clusters - 1)
    else:
        color_palette = sns.color_palette('bright', n_clusters)

    if soft:
        soft_clusters = all_points_membership_vectors(clusterer)
        cluster_colors = [color_palette[np.argmax(x)]
                          for x in soft_clusters]
    else:
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
    cluster_member_colors = cluster_colors#[sns.desaturate(x, p)
                            # for x, p
                            # in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_member_colors, color_palette, cluster_colors
    
    
def plot_vtk_data(mesh, cluster_scores: Dict, clusterer: HDBSCAN, legend: bool = True, save: bool = False,
                 figname: str = None, figpath: str = None, soft: bool = True) -> None:
                 
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    labels=clusterer.labels_

    cluster_member_colors, color_palette, cluster_colors = new_set_cluster_member_colors(clusterer, soft)

    if legend:
        if -1 in np.unique(labels):
            unique_colors = ((0.5, 0.5, 0.5), *tuple(color_palette))
        else:
            unique_colors = tuple(color_palette)
        cmap = colors.ListedColormap(unique_colors)
    
    # First a default plot with jet colormap
    p = pv.Plotter(notebook=False, off_screen=False)

    mesh['values'] = labels

    p.add_mesh(mesh, scalars='values', cmap=cmap)
    p.remove_scalar_bar()
    p.background_color = 'w'

    for cnt, (color, text) in enumerate(zip(cmap.colors, np.unique(labels))):
        score_dict_ori = get_score_dict(cluster_scores, text)
        score_dict = dict((k, v) for k, v in score_dict_ori.items() if float(v) >= 0)
        #idx = np.argwhere(mesh['values'] == text)
        #idx_sub = round(len(idx)/3)

        start    = [xmin, ymax + (ymax-ymin)/4, zmin + (cnt + 0) * 0.02]
        p.add_point_labels(points = [start],labels = ['{}'.format(score_dict)], 
                           text_color = color, font_size = 24, tolerance=0.01,
                           show_points = False, fill_shape = False, reset_camera=False)

        poly = pv.PolyData([xmin, ymax + (ymax-ymin)/6, zmin + (cnt + 0) * 0.02])
        poly["My Labels"] = [text]
        p.add_point_labels(poly, "My Labels", point_size=16, font_size=26, show_points = False, point_color = color, shape = 'rounded_rect', fill_shape = True, shape_color = color)
        
    p.camera_position = 'yz'

    p.store_image = True
    p.show()
    
    fig, ax = _set_plot_settings(figsize = (7.2, 4.45))
    ax.imshow(p.image)
    
    plt.axis('off')
    _save_fig(save, figname, figpath)

def plot_cluster_scores(cluster_scores: Dict, score_type: str = 'Mutual', save: bool = False, figname: str = None, figpath: str = None):

    for key in cluster_scores.keys():

        cluster_score = cluster_scores[key]

        if 'Mutual Information scores' in cluster_score.columns:
            x = 'Mutual Information scores'
        elif 'Correlation Coefficient' in cluster_score.columns:
            x = 'Correlation Coefficient'
            
        fig, ax = plt.subplots(figsize=[7, 7])
        palette = sns.color_palette('bright', 3)

        ax = sns.barplot(x=x, y='Variables',
                         hue='Synthetic variables', data=cluster_score, palette=palette)
        ax.set_xlabel(xlabel='{} for cluster: {}'.format(x, key), fontsize=17)
        ax.set_ylabel(ylabel='Original variables', fontsize=17)
        locs, labels = plt.yticks()
        new_yticks = fix_yticks(labels=labels)
        plt.yticks(locs, new_yticks, fontsize=17)
        plt.xticks(fontsize=17)
        plt.legend(fontsize=17)
        
        if save:
            _save_fig(save, figname + "{}".format(key), figpath)
        else:
            plt.show()

def plot_csv_data(coord, cluster_scores, clusterer: HDBSCAN, legend: bool = True, save: bool = False,
                 figname: str = None, figpath: str = None, soft: bool = True):
    n_clusters = np.size(np.unique(clusterer.labels_))
    cluster_member_colors, color_palette, cluster_colors = new_set_cluster_member_colors(clusterer, soft)

    if legend:
        if -1 in np.unique(clusterer.labels_):
            unique_colors = ((0.5, 0.5, 0.5), *tuple(color_palette))
        else:
            unique_colors = tuple(color_palette)

    fig, ax = plt.subplots(figsize=[7.2, 4.45])

    x = coord['Points:1']
    y = coord['Points:2']
    z = np.array(cluster_colors)

    X, Y, Z1, Z2, Z3 = sgrid(x, y, z, 1000j, 1000j)

    x_length = max(x) - min(x)
    y_length = max(y) - min(y)

    ccc = np.zeros((len(X), len(Y), 3))
    ccc[:, :, 0], ccc[:, :, 1], ccc[:, :, 2] = Z1.T, Z2.T, Z3.T
    ccc[np.isnan(ccc)] = 1.
    ax.imshow(ccc, origin="lower", extent=(min(x), max(x), min(y), max(y)))

    connectionstyle = "arc,angleA=-180,angleB=0,armA=30,armB=30,rad=5"

    for cnt, (color, text) in enumerate(zip(unique_colors, np.unique(clusterer.labels_))):
        score_dict = get_score_dict(cluster_scores, text)
        
        ax.add_patch(Circle((max(x) + 0.1 * x_length, 
                             min(y) + 0.1 * y_length + cnt * 0.12 * y_length), 
                             radius=y_length / 50, facecolor=color, 
                             edgecolor=color, clip_on=False))
                             
        ax.text(max(x) + 0.082 * x_length, 
                min(y) + 0.09 * y_length + cnt * 0.12 * y_length, 
                text, color="white", weight='bold', fontsize=13)

        idx = np.argwhere((z == np.array(unique_colors)[cnt]).all(axis=1))

        # xy position where arrow points to.
        eix = x[idx[round(len(idx) / 2)]]
        eiy = y[idx[round(len(idx) / 2)]]
        
        ax.annotate('{}'.format(score_dict), xy=(eix, eiy)
                     , xytext=(max(x) + 0.12 * x_length, min(y) + 0.1 * y_length + cnt * 0.12 * y_length)
                     , color=(color), weight='bold'
                     , fontsize=12, ha='left', va='center', clip_on=False
                     , arrowprops=dict(arrowstyle="-|>", connectionstyle=connectionstyle
                     , facecolor='black', ls="-", alpha=0.5, relpos=(-0.05, 0.5)))

    ax.set_xticks([min(x), (max(x)+min(x))/2, max(x)])
    ax.set_yticks([min(y), (max(y)+min(y))/2, max(y)])
    ax.set_xticklabels([-0.45, 0, 0.45], fontsize=16)
    ax.set_yticklabels([0, 0, 0.45], fontsize=16)

    ax.set_ylabel(r"$z/D$", fontsize=16)
    ax.set_xlabel(r"$r/D$", fontsize=16)
    ax.set_title('LESgrid', fontsize=16)
    fig.subplots_adjust(wspace=0.1, hspace=0.17)
    _save_fig(save, figname, figpath)
    
def sgrid(x, y, z, resX=50j, resY=50j):
    from scipy.interpolate import griddata

    a = 0.
    "Convert 3 column data to matplotlib grid"
    grid_x, grid_y = np.mgrid[min(x):max(x):resX, min(y):max(y):resY]

    method = "linear"

    xy = np.stack((x, y), axis=1)
    Z1 = griddata(xy, z[:, 0], (grid_x, grid_y), method=method, rescale=True)
    Z2 = griddata(xy, z[:, 1], (grid_x, grid_y), method=method, rescale=True)
    Z3 = griddata(xy, z[:, 2], (grid_x, grid_y), method=method, rescale=True)

    return grid_x, grid_y, Z1, Z2, Z3
    
def get_score_dict(cluster_scores, cluster_label):
    cluster_score = cluster_scores[cluster_label]
    
    if 'Mutual Information scores' in cluster_score.columns:
        score = cluster_score[cluster_score['Synthetic variables'] == 'UMAP X1-axis'].drop(columns='Synthetic variables').iloc[:3]
        score_dict = dict(zip(score['Variables'], score['Mutual Information scores'].map('{:,.2f}'.format)))
    elif 'Correlation Coefficient' in cluster_score.columns:
        score = cluster_score[cluster_score['Synthetic variables'] == 'Correlation'].drop(columns='Synthetic variables').iloc[:3]
        score_dict = dict(zip(score['Variables'], score['Correlation Coefficient'].map('{:,.2f}'.format)))
        
    return score_dict

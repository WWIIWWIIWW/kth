import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from .plot import plot_cluster_scores
from .plot import new_set_cluster_member_colors
from .dimred import scale_data
from scipy.stats import linregress
    
from typing import Union, Dict

def get_mi_scores(X, y, scale: bool = False):
    mi_scores = mutual_info_regression(X, y)
    if scale:
        mi_scores /= np.max(mi_scores)
    return mi_scores
    
def get_linear_scores(X, y, scale: bool = False):
    score = linregress(X, y.values.reshape(-1, 1).flatten()).rvalue
    return score

def get_cluster_mi_scores(data, clusterer, embedding, cluster_num: Union[str, int] = 'All', scale: bool = False, scaler: str = 'StandardScaler', feature: str = 'All', flag_print: bool = False) -> Dict:
    if scale:
        data = scale_data(data, scaler, feature)

    clustered_data = data.copy()
    clustered_data['clusters'] = clusterer.labels_
    
    if embedding.shape[1] == 3:
            clustered_data[['Var_X1', 'Var_X2', 'Var_X3']] = embedding
    elif  embedding.shape[1] == 2:
            clustered_data[['Var_X1', 'Var_X2']] = embedding


    if type(cluster_num) is int:
        clustered_data = clustered_data[(clustered_data['clusters'] == cluster_num)]

    cluster_mi_scores = {}
    for cluster in np.sort(clustered_data['clusters'].unique()):
    
        cluster_target = clustered_data[(clustered_data['clusters'] == cluster)].copy()

        cluster_target.drop(columns='clusters', inplace=True)

        if embedding.shape[1] == 3:
            X = cluster_target.drop(columns=['Var_X1', 'Var_X2', 'Var_X3'])
            y = cluster_target[['Var_X1', 'Var_X2', 'Var_X3']]

            df_MI = pd.DataFrame(
                {
                    'Variables': X.columns,
                    'UMAP X1-axis': get_mi_scores(X, y['Var_X1'], scale=scale),
                    'UMAP X2-axis': get_mi_scores(X, y['Var_X2'], scale=scale),
                    'UMAP X3-axis': get_mi_scores(X, y['Var_X3'], scale=scale)
                })

            cluster_mi_score = df_MI.melt(id_vars=['Variables'], 
                                        value_vars=['UMAP X1-axis', 'UMAP X2-axis', 'UMAP X3-axis'],
                                        var_name='Synthetic variables', 
                                        value_name='Mutual Information scores')
        elif  embedding.shape[1] == 2:
            X = cluster_target.drop(columns=['Var_X1', 'Var_X2'])
            y = cluster_target[['Var_X1', 'Var_X2']]

            df_MI = pd.DataFrame(
                {
                    'Variables': X.columns,
                    'UMAP X1-axis': get_mi_scores(X, y['Var_X1'], scale=scale),
                    'UMAP X2-axis': get_mi_scores(X, y['Var_X2'], scale=scale)
                })

            cluster_mi_score = df_MI.melt(id_vars=['Variables'], 
                                        value_vars=['UMAP X1-axis', 'UMAP X2-axis'],
                                        var_name='Synthetic variables', 
                                        value_name='Mutual Information scores')

        cluster_mi_score.sort_values('Mutual Information scores',
                                     ascending=False, inplace=True,
                                     ignore_index=True)

        if flag_print:
            print(f'Mutual Information scores for cluster {cluster}: \n')
            print(cluster_mi_score)
            print('\n')
            
        cluster_mi_scores[cluster] = cluster_mi_score

    return cluster_mi_scores
    
"""    
def get_cluster_scores(data, clusterer, embedding, cluster_num: Union[str, int] = 'All', scale: bool = False, scaler: str = 'StandardScaler', feature: str = 'All', flag_print: bool = False) -> Dict:
    
    if scale:
        data = scale_data(data, scaler, feature)

    clustered_data = data.copy()
    clustered_data['clusters'] = clusterer.labels_
    
    if embedding.shape[1] == 3:
            clustered_data[['Var_X1', 'Var_X2', 'Var_X3']] = embedding
    elif  embedding.shape[1] == 2:
            clustered_data[['Var_X1', 'Var_X2']] = embedding


    if type(cluster_num) is int:
        clustered_data = clustered_data[(clustered_data['clusters'] == cluster_num)]

    cluster_mi_scores = {}
    for cluster in np.sort(clustered_data['clusters'].unique()):
    
        cluster_target = clustered_data[(clustered_data['clusters'] == cluster)].copy()

        cluster_target.drop(columns='clusters', inplace=True)

        if embedding.shape[1] == 3:
            X = cluster_target.drop(columns=['Var_X1', 'Var_X2', 'Var_X3'])
            y = cluster_target[['Var_X1', 'Var_X2', 'Var_X3']]
            
            corrX1 = []
            corrX2 = []
            corrX3 = []
            for species in X.columns:
                scoreX1 = get_linear_scores(X[species], y['Var_X1'])
                scoreX2 = get_linear_scores(X[species], y['Var_X2'])
                scoreX3 = get_linear_scores(X[species], y['Var_X3'])
                corrX1.append(scoreX1)
                corrX2.append(scoreX2)
                corrX3.append(scoreX3)
                
            df_MI = pd.DataFrame(
                {
                    'Variables': X.columns,
                    'UMAP X1-axis': corrX1,
                    'UMAP X2-axis': corrX2,
                    'UMAP X3-axis': corrX3
                })

            cluster_mi_score = df_MI.melt(id_vars=['Variables'], 
                                        value_vars=['UMAP X1-axis', 'UMAP X2-axis', 'UMAP X3-axis'],
                                        var_name='Synthetic variables', 
                                        value_name='Mutual Information scores')
        elif  embedding.shape[1] == 2:
            X = cluster_target.drop(columns=['Var_X1', 'Var_X2'])
            y = cluster_target[['Var_X1', 'Var_X2']]

            corrX1 = []
            corrX2 = []
            for species in X.columns:
                scoreX1 = get_linear_scores(X[species], y['Var_X1'])
                scoreX2 = get_linear_scores(X[species], y['Var_X2'])
                corrX1.append(scoreX1)
                corrX2.append(scoreX2)
                
            df_MI = pd.DataFrame(
                {
                    'Variables': X.columns,
                    'UMAP X1-axis': corrX1,
                    'UMAP X2-axis': corrX2
                })

            cluster_mi_score = df_MI.melt(id_vars=['Variables'], 
                                        value_vars=['UMAP X1-axis', 'UMAP X2-axis'],
                                        var_name='Synthetic variables', 
                                        value_name='Mutual Information scores')

        cluster_mi_score.sort_values('Mutual Information scores',
                                     ascending=False, inplace=True,
                                     ignore_index=True)

        if flag_print:
            print(f'Mutual Information scores for cluster {cluster}: \n')
            print(cluster_mi_score)
            print('\n')
            
        cluster_mi_scores[cluster] = cluster_mi_score

    return cluster_mi_scores
"""

def get_cluster_scores(data, clusterer, embedding, cluster_num: Union[str, int] = 'All', scale: bool = False, scaler: str = 'StandardScaler', feature: str = 'All', flag_print: bool = False) -> Dict:
    
    if scale:
        data = scale_data(data, scaler, feature)

    clustered_data = data.copy()
    clustered_data['clusters'] = clusterer.labels_
    
    dim = embedding.shape[1]
    #Var_X1, Var_X2 ...
    clustered_data[['Var_X' + str(i+1) for i in range(dim)]] = embedding
     
    if type(cluster_num) is int:
        clustered_data = clustered_data[(clustered_data['clusters'] == cluster_num)]

    cluster_scores = {}
    
    for cluster in np.sort(clustered_data['clusters'].unique()):
    
        cluster_target = clustered_data[(clustered_data['clusters'] == cluster)].copy()

        cluster_target.drop(columns='clusters', inplace=True)

        if embedding.shape[1] == 3:

            X = cluster_target.drop(columns=['Var_X1', 'Var_X2', 'Var_X3'])
            y = cluster_target[['Var_X1', 'Var_X2', 'Var_X3']]
            Y = (y['Var_X1']+ y['Var_X2']+ y['Var_X3']) / 3
        elif embedding.shape[1] == 2:

            X = cluster_target.drop(columns=['Var_X1', 'Var_X2'])
            y = cluster_target[['Var_X1', 'Var_X2']]
            Y = (y['Var_X1']+ y['Var_X2']) / 2
            
        corr = []

        for species in X.columns:
            score = get_linear_scores(X[species], Y)
            corr.append(score)
                
        df_MI = pd.DataFrame(
            {
                'Variables': X.columns,
                'Correlation': corr,
            })

        cluster_score = df_MI.melt(id_vars=['Variables'], 
                                    value_vars=['Correlation'],
                                    var_name='Synthetic variables', 
                                    value_name='Correlation Coefficient')

        cluster_score.sort_values('Correlation Coefficient',
                                  ascending=False, inplace=True,
                                  ignore_index=True)

        if flag_print:
            print(f'Correlation Coefficient for cluster {cluster}: \n')
            print(cluster_score)
            print('\n')
            
        cluster_scores[cluster] = cluster_score

    return cluster_scores

import numpy as np
from scipy.spatial.distance import cdist
import pandas, random


# %% [markdown]
# # Creating recommendation based on a song

# %%
### Predicting and creating playlist around a track

# Meant for the base track or the stat you want to scale
def scale_predict(stats,centroids_manual,scaler):
    stats_scaled=scaler.transform(stats)
    label = np.argmin(cdist(stats_scaled,centroids_manual),axis=1)
    return label, stats_scaled


# %%

def sort_distances(cluster,base_stats_scaled,cluster_data_scaled):
    cluster["Distances"]=cdist(cluster_data_scaled,np.mean(base_stats_scaled,0).reshape(1,-1))
    cluster=cluster.sort_values("Distances")
    return cluster

# %%

def sort_closest(cluster,mean_stats_scaled,cluster_data_scaled):

    cluster.reset_index(drop=True,inplace=True)
    try:
        data=np.array(cluster_data_scaled)
    except:
        raise ValueError("cluster_data_scaled has to be an array or list, not ", type(cluster_data_scaled))
    current=np.mean(mean_stats_scaled,0).reshape(1,-1)
    indexes=[]
    while len(indexes)<len(data):
        min_idx=np.argmin(cdist(data,current))    
        current=data[min_idx,:].reshape(1,-1).copy()
        # Mutate `data` to mark visited points
        data[min_idx,:]=10000

        indexes.append(min_idx)
    return cluster.loc[indexes,:]


# %%
def recommend(base, song_data, centroids, values, scaler):
    label, base_stats = scale_predict(base[values],centroids,scaler)
    cluster=song_data[song_data["Label 2"]==label]
    cluster_stats=scaler.transform(cluster[values])
    sorted_cluster=sort_distances(cluster,base_stats,cluster_stats)

    recs_playlist=pandas.concat([base,sorted_cluster])
    return recs_playlist


# %%
from playlist.manager import read_csv, split_genres, count_genres_in_labels, rotate_until, Playlist
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random

# # Manual Tagging

# %%
def explode_genre(data: pandas.DataFrame):
    new_playlist=data.copy()

    # genres=split_genres(new_playlist)
    # new_playlist['Genres']=genres

    df = new_playlist.explode('Genres')
    df.set_index("Genres",inplace=True)
    # df[["Track ID","Genres"]]
    return df

def _s(df):
    grouped=df.groupby("Genres")
    return grouped


# %%
# ## Finding rare genres
# %%
### From how many occurances can I keep a genre to keep half?

def find_genre_treshold(grouped,x):
    if x<0 or x>1:
        raise ValueError(f"x has to be between 0.0 and 1.0, got {x}")
    histo=[]
    n_genres=len(grouped)
    for idx in range(int(n_genres)):
        genres_rare=[]
        genres_common=[]
        for genre in grouped:
            if len(genre[1])<=idx:
                # print(f"There is only {len(genre[1])} {genre[0]} found.")
                genres_rare.append(genre[0])
            else:
                genres_common.append(genre[0])
        # print(f'Less than or equal to {i} genres: {len(genres_rare)}, more than {i}: {len(genres_common)}')
        histo.append(len(genres_rare))
    
    # How many can I save if a supposed subgenre or bigger category is available?
    treshold=_find_remaining_treshold(grouped,histo,x)

    return treshold

def _find_remaining_treshold(grouped,histo,x):
    n_genres=len(grouped)
    remaining_histo=[]
    for idx in range(n_genres):
        genres_common=[]
        for genre in grouped:
            if len(genre[1])>idx:
                genres_common.append(genre[0])
        saveable=0
        for genre in grouped:
            if len(genre[1])<=idx:
                # print(f"There is only {len(genre[1])} {genre[0]} found.")
                # String manipulation
                if any([common in genre[0] for common in genres_common]) or any([genre[0] in common for common in genres_common]):
                    # print("!! There is a solution")
                    saveable+=1
        x=n_genres-histo[idx]+saveable
        remaining_histo.append(x)
        # print(f'You can save {saveable} genres out of {histo[i]} at {i}. Percentage: {saveable/(histo[i]+.000001)*100:.2f}%. Remaining genres: {x}')

    for idx,h in enumerate(remaining_histo):
        if h<x*n_genres:
            treshold=idx-1
            break

    print(treshold,": ",remaining_histo[treshold])
    print(remaining_histo[treshold]/n_genres)

    return treshold

# %%
# get common vs rare genres based on treshold
def get_genres_at_treshold(grouped,tresh):
    genres_rare=[]
    genres_common=[]
    safe=0
    saveable=0
    for genre in grouped:
        if len(genre[1])<=tresh:
            # print(f"There is only {len(genre[1])} {genre[0]} found.")
            genres_rare.append(genre[0])
            if any([common in genre[0] for common in genres_common]) or any([genre[0] in common for common in genres_common]):
                # print("!! There is a solution")
                saveable+=1
        else:
            genres_common.append(genre[0])
            safe+=1
    print(f'Less than or equal to {tresh} genres: {len(genres_rare)}, more than {tresh}: {len(genres_common)}')
    print(f"Lost: all - safe - saveable = {len(grouped)} - {safe} - {saveable} = {len(grouped)-safe-saveable}")
    return genres_common, genres_rare, safe,saveable

# %%
# modifying data
def drop_below_tresh(grouped, i, genres_common):
    dfs=[]
    print(f'Current cutoff: {i}')
    for genre in grouped:
        if len(genre[1])<=i:
            if any([common in genre[0] for common in genres_common]):
                locate=np.where(np.array([common in genre[0] for common in genres_common]))[0].tolist()
                changed_genre=[genres_common[l] for l in locate][-1]
                # print('1.',genre[0],': ',changed_genre)
                genre[1]["Genres"]=changed_genre
                dfs.append(genre[1])
                continue
        
            if any([genre[0] in common for common in genres_common]):
                locate=np.where(np.array([genre[0] in common for common in genres_common]))[0].tolist()
                changed_genre=[genres_common[l] for l in locate]
                # print('2.',genre[0],': ',changed_genre)
                # print(len(genre[1]),"needed")
                # print(len(changed_genre),"genres")
                to_assign=[changed_genre for _ in genre[1].values]
                # print(len(to_assign))
                genre[1]["Genres"]=to_assign
                dfs.append(genre[1])
                
        else:
            dfs.append(genre[1])
    print(len(dfs))

    new_df=pandas.concat(dfs)

    print(len(new_df))

    new_df = new_df.explode('Genres')
    new_df.reset_index(drop=True,inplace=True)

    print(len(new_df))

    return new_df


# %%
## WORKFLOW

def preprocess(data: pandas.DataFrame):
    values=[]

    numeric_columns = data.select_dtypes(include=np.number).columns
    print(numeric_columns)    
    
    # Scale data stats
    scaler= MinMaxScaler(feature_range=(0, 1))
    stats_scaled=scaler.fit_transform(data[numeric_columns])
    # and replace
    data[numeric_columns]=stats_scaled
    return data, scaler

def process(data: pandas.DataFrame,values:list,scaler:MinMaxScaler,n_clusters,seed):
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters,random_state=seed)
    kmeans.fit(data[values])

    # explode song data based on genres
    genre_df=explode_genre(data)
    genre_stats=genre_df[values]
    labels=kmeans.predict(genre_stats)
    # from genres get p_matrix 
    assert len(labels) == len(genre_stats), "Mismatch in genre labels"
    genre_labels_sorted=sorted(list(tuple(zip(genre_stats.index,labels))),key=lambda x : x[1])
    genre_to_labels=group_labels_by_genre(genre_labels_sorted)
    probabilities, most_common= calc_genre_distributions(genre_to_labels,genre_labels_sorted,n_clusters)
    item_to_genres=group_genres_by_labeL(most_common)
    # The Matrix
    p_matrix=pandas.DataFrame(probabilities,index=genre_to_labels.keys())


    ## dealing with songs from the results
    labeled_data,nan_data=train_songs(data,p_matrix,values,kmeans)
    return labeled_data, scaler, kmeans

# %%
# ### Fit, tranform, predict

# %%

# Call this for training the playlist
def kmeans_train(stats: pandas.DataFrame,scaler:MinMaxScaler,kmeans:KMeans):
    stats_scaled=scaler.fit_transform(stats) # For songs
    kmeans.fit(stats_scaled)
    return scaler, kmeans

# Call this for general transform after
def kmeans_scale_predict(stats: pandas.DataFrame,scaler:MinMaxScaler,kmeans:KMeans):
    stats_scaled=scaler.transform(stats) # For genres
    labels=kmeans.predict(stats_scaled)

    return labels


# %%
# ### Calc Matrix and various dicts

# Create dictionary: {genre: [list of associated labels]}
def group_labels_by_genre(genre_labels_sorted):
    genre_to_labels={l[0]:[] for l in genre_labels_sorted}
    for key,value in genre_labels_sorted:
        genre_to_labels[key].append(value)
    return genre_to_labels


def calc_genre_distributions(genre_to_labels,genre_labels_sorted,n_clusters):
    probabilities=np.zeros((len(genre_to_labels.keys()),n_clusters))
    most_common={l[0]:[] for l in genre_labels_sorted}

    # For each genre, calculate how its items are distributed across clusters
    for idx,genre_name in enumerate(genre_to_labels):
        cluster_label=genre_to_labels[genre_name]
        cluster_counts = [0]*n_clusters
        for label in cluster_label:
            # cluster_counts[label]=cluster_label.count(label)
            cluster_counts[label]+=1
        probabilities[idx]=np.array(cluster_counts)/sum(cluster_counts)
        listy=probabilities[idx]
        most_common[genre_name]=np.argwhere(listy == np.amax(listy)).transpose()[0]
        # print(f'{key}, {most_common[key]}, {listy}')#, *(f"{value/sum(unique_values):g}" if value!=0 else "0" for value in unique_values))
    return probabilities, most_common

def group_genres_by_labeL(most_common):
    item_to_genres = {elem: [] for l in most_common for elem in most_common[l]}
    for key in most_common:
        for elem in most_common[key]:
            item_to_genres[elem].append(key)

    # Create reverse mapping (cluster -> genres) 
    item_to_genres={k: v for k, v in sorted(item_to_genres.items(), key=lambda item: item[0])}
    return item_to_genres



# %%
# ### Calc the best matching cluster for each song from data
# The crust of the algorithm
def train_songs(data:pandas.DataFrame,p_matrix,values,kmeans:KMeans):
    w_song=1.0 #weight of song label
    # 1. Assigns the most relevant label (column with the highest sum) to each index where the genre exists.
    label_list=[]
    index_list=[]
    nan_index=[]
    print(f"Shape of p: {np.shape(p_matrix)}")
    centroids=np.array(kmeans.cluster_centers_)
    # print(f"Shape of centroids: {np.shape(centroids)}")

    for idx,data_row in data.iterrows():
        genre=data_row["Genres"]
        try:
            label=kmeans.labels_[idx]
        except KeyError:
            print("Something is not ok.")
            pass
        
        # If genre exists
        if type(genre)==list:
            index_list.append(int(idx))
            # Finds the column with the highest mean across the selected genre rows in p_matrix.
            p_new=p_matrix.loc[genre].copy().mean(axis=0)
            row_values=np.reshape(data_row[values].to_numpy(),(1,-1)).astype(np.float64)
            # print(np.shape(row_values))
            label_fuzzy=cdist(centroids,row_values)
            p_new=p_new+w_song*label_fuzzy[:,0]/sum(label_fuzzy)
            
            label_list.append(p_new.argmax())
        else:
            nan_index.append(idx)

    data.loc[index_list,"Label 2"]=label_list

    # 2. Compute centroids from known labels

    group_label=data.groupby("Label 2")
    centroids_manual=group_label[values].mean()

    # 3. Handle NaN values
    nan_data=data.loc[nan_index].copy()
    values_manual=nan_data.loc[:,values]

    ### Assign to closest
    nan_data["Label 2"]=np.argmin(cdist(values_manual,centroids_manual),axis=1)

    labeled_data=pandas.concat([data.dropna(subset="Label 2"), nan_data])
    print(f"{len(data)} -> {len(labeled_data)}")

    return labeled_data, nan_data



# # %% [markdown]
# # ## Generating names and saving all

# %%
def make_names(list,length):
    names=[]
    for elem in list:
        out=', '.join([elem[idx][0] for idx in range(min(length,len(elem)))])
        names.append(out)
    return names

# names=make_names(counted_list)

# %%

def sort_playlist(data,n_clusters,values,idx=None):
    playlist=Playlist(data)
    playlist.set_clusters(f=values,no_of_clusters=n_clusters)

    # sort by familiarity to mean
    centroid_all=np.reshape(playlist.features.mean(axis=0),(1,-1))
    centroid_sort=np.array(playlist.centroids)
    dists=cdist(centroid_all,centroid_sort,"euclidean")[0]
    order=np.argsort(dists)

    # sort by popularity
    # centroids_sort=np.array(playlist.centroids)
    # order= np.argsort(centroids_sort[:,idx])[::-1]
    
    # most popular first
    # order=list(range(n_clusters))
    # order=rotate_until(order,np.argmax(playlist.centroids, axis=0)[idx])


    return playlist.process2(order).songs
    
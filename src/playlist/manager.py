# %% 
# # playlist manager
import time
import matplotlib.pyplot as plt

import_all=time.time()
import pandas
print(f"Time to import pandas: {time.time()-import_all}")

import numpy as np

import1=time.time()
from sklearn.cluster import KMeans
print(f"Time to import KMeans: {time.time()-import1}")

import copy
from scipy.spatial.distance import cdist
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from collections import Counter
print(f"Time to import all: {time.time()-import_all}")

# %%
#for order properties
class Order:
    def __init__(self, n_clusters, order = None):
        self.n=n_clusters
        self.order=order

    
    def count_transition(self):
        counts={transition: 0 for transition in self.transitions}
        reverse=[]
        for start, end in zip(self.order, self.order[1:]):  # Using zip to iterate adjacent pairs
            if (start, end) in counts:  # Check if the transition exists in counts
                counts[start, end] += 1
                reverse.append(False)
            if (end, start) in counts:  # Check if the transition exists in counts
                counts[end, start] += 1
                reverse.append(True)
        return counts
    
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        if order is None:
            self._order=[]
            return
        if max(order)>self.n-1 or min(order)<0:
            raise ValueError("'order' can't contain non-existent cluster ("+str(max(order))+")")
        self._order=order

        self.transitions=self.set_transitions()

    @property
    def crossings(self):
        # Order/crossings pairing
        data2 = pandas.DataFrame({
            'Order': self.order, 
            'Crossings': [1] + [2] * (len(self.order)-2) + [1]  # List where first and last elements are 1, others are 2
        })
        
        crossings = []
        
        # Loop through clusters and compute sum
        for cluster in set(self.order):
            sum_value = data2.loc[data2['Order'] == cluster, 'Crossings'].sum()
            crossings.append(int(sum_value))
        return crossings

    def set_transitions(self):
        pairs=set()
        for start, end in zip(self.order,self.order[1:]):
            if (start,end) not in pairs or (end,start) not in pairs:
                pairs.add((start,end)) if start<end else pairs.add((end,start))
        pairs=list(pairs)
        pairs.reverse()
        return pairs

    def from_transitions(self, transitions):
        pass
# %%
class Playlist:
    def __init__(self, data: pandas.DataFrame = None, no_of_clusters: int = None, order = None):
        if data is None:
            self.songs = pandas.DataFrame()
        else:
            self.songs = data.reset_index()
        self.order=Order(no_of_clusters,order)
        self.transitions=[]
        self.features=np.array([])
        self.members=[]
        self.no_of_clusters=no_of_clusters
        self.centroids=None

    
    def set_clusters(self,f=['Energy','Danceability','Valence'], no_of_clusters=3, init=None):
        
        if init is None:
            init='k-means++'
            n_init='auto'
        else:
            n_init=1
        #     raise ValueError("You need to define init if no_of_clusters is not 3")
        if init is None and no_of_clusters == 3 and len(f)==3:
            init=[ [0, 0, 0],[1, .5, 1],[0.5, 1, 1]]
            n_init=1
        self.no_of_clusters=no_of_clusters # should this be set here?
        self.f=f
        kmeans = KMeans(n_clusters=no_of_clusters,init=init,random_state=42,n_init=n_init)
        self._scaler= MinMaxScaler(feature_range=(0, 1))
        self.features=self._scaler.fit_transform(self.songs[self.f])
        kmeans.fit(self.features)
        # self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        self.labels, self.members=self.get_fuzzy_labels()
        self.songs["Label"]=self.labels
        for idx,m in enumerate(self.members):
            self.songs[str(idx)]=m

    def get_fuzzy_labels(self,centroids=None):
        if centroids is None:
            centroids=self.centroids
        
        # Calculate distances to centroids
        distances = cdist(self.features, centroids, metric='euclidean')
        # Convert distances to fuzzy membership scores
        # Using a Gaussian kernel
        sigma = distances.std()  # Or tune this manually
        fuzzy_memberships = np.exp(-distances**2 / (2 * sigma**2))
        fuzzy_memberships /= fuzzy_memberships.sum(axis=1, keepdims=True)  # Normalize
        labels_fuzzy=np.argmax(fuzzy_memberships, axis=1)
        return labels_fuzzy, fuzzy_memberships.transpose()

    def set_order(self, order):# where should this be called? Init?
        self.order.order=order

    def divide_random(self,seed=32628592046396213140428242853833564776):
        rng=np.random.default_rng(seed)
        crossings=np.array(self.order.crossings)
        divide=crossings[self.songs["Label"].to_numpy()]
        # self.songs["Division"]=[ rng.integers(d)+1 for d in divide]
        # divisions=self.songs.groupby(["Label", "Division"])
        return [ rng.integers(d)+1 for d in divide]

    def divide_remainder(self):
        crossings=self.order.crossings
        divisions=[0]*len(self.songs)
        divisions_label=self.songs.groupby(["Label"])
        for div in divisions_label:
            key=div[0][0]
            for idx,elem in enumerate(div[1].index):
                divisions[elem]=idx % crossings[key] +1
        return divisions

    def set_division(self,division_function):
        self.songs["Division"]=division_function()

    def sort(self):
        #variables needed: self.no_of_clusters, self.order.order, self.songs["Division"]
        divisions=self.songs.groupby(["Label", "Division"])
        #Create subsets from the 2 clusters that are next to each other in the order
        #I created a Division column, so I don't have worry about it when selecting, I just add 1 to the division when one is used from the cluster
        #order[order_idx], d cluster xth order 
        #d cluster current order +1
        #
        #order[order_idx+1], d cluster order_idx+1 order
        #d cluster order_idx+1 order +1
        current_division_index = [1] * self.no_of_clusters
        sorted_clusters =[]
        
        epsilon= 0.5 #smoothing
        for order_idx in range(len(self.order.order)-1):
            #Clusters: start, end

            current_cluster =self.order.order[order_idx ]
            next_cluster =self.order.order[order_idx +1]
            #

            
            current_cluster_data = divisions.get_group((current_cluster, current_division_index[current_cluster]))
            
            if current_cluster==next_cluster:# add different sorting if the 2 clusters next to each are the same
                next_cluster_data = divisions.get_group((next_cluster, current_division_index[next_cluster] + 1))
            else:
                next_cluster_data = divisions.get_group((next_cluster, current_division_index[next_cluster]))
            
            merged_data = pandas.concat([current_cluster_data, next_cluster_data])
            
            ##Normalize
            merged_data["Diff"] = merged_data[str(current_cluster)] / (
                merged_data[str(current_cluster)] + merged_data[str(next_cluster)] + epsilon
                )            
            #ordered_clusters[order_idx ]["Diff"]=ordered_clusters[order_idx ][str(start)]-ordered_clusters[order_idx ][str(end)]
            ##Sort now
            sorted_clusters.append(merged_data.sort_values("Diff", ascending=False))

            current_division_index[current_cluster]+=1
            current_division_index[next_cluster]+=1

        # Concatenate all sorted songs into a final DataFrame
        ordered_data = pandas.concat(sorted_clusters, ignore_index=True)   
        
        return ordered_data
        #Order each subset based on how much they belong to the 2 clusters
        #Ignore the clusters we aren't interested in
        #Normalize fuzzy distances to 1
        #songs[
        #Sort from order[order_idx] to order[order_idx+1]

    def process(self,order,division_method=None,sorting_method=None,in_place=False):
        self.order=Order(self.no_of_clusters,order)
        if division_method==None:
            division_method=self.divide_remainder
        if sorting_method==None:
            sorting_method=self.sort
        self.set_division(division_method)

        ordered_data=self.sort()
        if in_place:
            self.songs=ordered_data
            return None
        else:
            new_playlist=copy.deepcopy(self)
            new_playlist.songs=ordered_data
            return new_playlist

    def process2(self,order,in_place=False):
        self.order=Order(self.no_of_clusters,order)
        self.songs["Diff"]=0

        filtered_order=self._find_missing_and_gen_new_order()
        if filtered_order:
            self.order.order=filtered_order
            
        else:
            print("Nothing is missing.")
        
        # Calculate which transition member each song belongs to
        transition_members=self.calc_members()

        # Sort each transition
        transitions_sorted={}
        for transition_member,transition in zip(transition_members.values(),self.order.transitions):
            transitions_sorted[transition]=self.transition_sort(transition_member,transition)

        # Count
        counts=self.order.count_transition()
        
        
        # TODO: make some kind of improvement to remainder method (eg. past values influence which one to choose from)
        # Divide each transition into groups based on how many times it occurs in order
        divisions=[]
        for key in transitions_sorted:
            # Get a list back with the size of the amount of divisions
            division=self.set_division2(transitions_sorted[key],counts[key],key)
            divisions=divisions+division

        # Add the results together (and reverse if needed)
        ordered_data=self.add_in_order(divisions)

        # Save
        if in_place:
            self.songs=ordered_data
            self.features=self._scaler.transform(self.songs[self.f])
            
        else:
            new_playlist=copy.deepcopy(self)
            new_playlist.songs=ordered_data
            new_playlist.features=self._scaler.transform(new_playlist.songs[self.f])
            return new_playlist
        

    def _filter_order(self,missing_transitions):
        filtered_order = deque([self.order.order[0]])
        for i in range(1, len(self.order.order)):
            prev = filtered_order[-1]
            curr = self.order.order[i]
            if prev==curr:
                continue
            if (prev, curr) in missing_transitions or (curr, prev) in missing_transitions:
                continue
            filtered_order.append(curr)

        return list(filtered_order)
        
    def _find_valid_transitions(self,max12):
        tops=set()
        for idx, scores in enumerate(max12):
            top2=tuple(sorted(scores[:2].tolist()))
            if top2 not in self.order.transitions:
                #find xth biggest pair and return first
                top2=self._find_xth_biggest_pair(scores,self.order.transitions)
            tops.add(top2)
        return tops
        
    def _find_xth_biggest_pair(self,scores, transitions):
        for first in range(len(scores)-1):
            for second in range(first+1,len(scores)):
                top2=tuple(sorted([scores[first],scores[second]]))
                if top2 in transitions:
                    return top2
        return None

    def _find_missing_and_gen_new_order(self):
        labels=[str(r) for r in range(self.no_of_clusters)]
        max12=np.argsort(self.songs[labels].to_numpy(),axis=1)[:,::-1]
        tops=self._find_valid_transitions(max12)
        if any([tr not in tops for tr in self.order.transitions]):
            missing_transitions=[tr for tr in self.order.transitions if tr not in tops]
            print("Missing transitions: ",len(missing_transitions),missing_transitions)
            
            filtered_order=self._filter_order(missing_transitions)
            print("Filtered order: ",filtered_order)
            return filtered_order
        else:
            return None

    def calc_members(self):
        ## in some cases some clusters are missing in a transition resulting in 'jerkier' sorting
        transitions=self.order.transitions

        labels=[str(r) for r in range(self.no_of_clusters)]
        max12=np.argsort(self.songs[labels].to_numpy(),axis=1)[:,::-1]
    
        zip_transitions={transition: [] for transition in self.order.transitions}

        for idx, scores in enumerate(max12):
            top2=tuple(sorted(scores[:2].tolist()))
            if top2 not in self.order.transitions:
                #find xth biggest pair and return first
                top2=self._find_xth_biggest_pair(scores,self.order.transitions)
                # print("top2 changed!")
                # print(idx,top2)

            zip_transitions[top2].append(idx)

        # print("Found transitions after: ",len(tops),tops)

        
        

        transition_membership_df ={transition: self.songs.loc[zip_transitions[transition]] for transition in zip_transitions}
        
        return transition_membership_df

    
    def transition_sort(self, transition_member: pandas.DataFrame, transition):
        epsilon= 0.5 #smoothing
        transition_member["Diff"]=transition_member[str(transition[0])]/  \
                        (transition_member[str(transition[0])]+transition_member[str(transition[1])]+epsilon)
        return transition_member.sort_values("Diff",ascending=False,ignore_index=True)


    def set_division2(self, transition_members,count, transition):
        # print(transition,count)
        # print(transition_members.index)
        transition_members["Division"]=[i % count for i in transition_members.index]
        divisions_per_transition=[]
        grouped=transition_members.groupby('Division')
        for division in grouped:
            divisions_per_transition.append([transition,division[1]])
            # print(transition[0],': ',len(division[1][division[1]["Label"]==transition[0]]))
            # print(transition[1],': ',len(division[1][division[1]["Label"]==transition[1]]))
        return divisions_per_transition

    def add_in_order(self,divisions):
        ordered_list=pandas.DataFrame()
        for start, end in zip(self.order.order,self.order.order[1:]):
            # print((start,end))
            for idx,div in enumerate(divisions):
                if div[0]==(start,end):
                    rmd=div[1].copy()
                    # print("Forward match found!, ",div[0])            
                    ordered_list=pandas.concat([ordered_list,rmd])
        
                    del divisions[idx]
                    break
                if div[0]==(end,start):
                    rmd=div[1].copy()
                    rmd=rmd.iloc[::-1]
                    # print("Reverse match found!, ",div[0])
                    ordered_list=pandas.concat([ordered_list,rmd])
                    del divisions[idx]
                    break
        # print(ordered_list)
        # ordered_list_pd=pandas.DataFrame(ordered_list,columns=self.songs.columns)
        return ordered_list

    def sort_rec(self, track):
        features=self._scaler.transform(track[self.f])
        distances=cdist(self.features, features, metric='euclidean')
        self.songs["Distance"]=distances
        return self.songs.sort_values("Distance")
        
    #todo: need to handle some cases for plots
    def plot3d(self,i=None,lines=False,index_text=False,features=None):
         

        if not features:
            xyz=self.f[0:3]
            features=self.features
        else:
            if len(features)!= 3:
                raise ValueError('features has to be len=3', features)
            ids=np.array([self.f.index(feat) for feat in features])
            xyz=features
            features=self.features[:,ids]
        if i:
            points=features.transpose()[0:3,0:i]
        else:
            points=features.transpose()[0:3,:]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting the points, colored by cluster labels
        scatter = ax.scatter(points[0], points[1], points[2], c=range(len(points[0])), cmap='viridis',  s=50)
        # if self.centroids.all():
        #     scatterc=ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
        #                         c='red', s=100, marker='x', alpha=1) 

        # Adding labels and title
        ax.set_xlabel(str(xyz[0]))
        ax.set_ylabel(str(xyz[1]))
        ax.set_zlabel(str(xyz[2]))
        ax.set_title('3D Scatter Plot of Songs')

        # Adding text to points
        if index_text:
            for j, point in enumerate(points.transpose()):
                ax.text(point[0],point[1],point[2],str(j))

        # Adding line
        if lines:
            for i in range(len(points[0]) - 1):
                ax.plot([points[0][i], points[0][i+1]],
                            [points[1][i], points[1][i+1]],
                            [points[2][i], points[2][i+1]], c='gray')

        # Adding text to centroids
        # if centroids.all():
        #     for idx, xyz in enumerate(centroids):
        #         ax.text(xyz[0],xyz[1]+0.02,xyz[2]-0.05, str(idx), size='xx-large')

        # Add a color bar
        plt.colorbar(scatter)
        return fig
        
    def plot3d2(self,features=None):
        import matplotlib.pyplot as plt

        if not features:
            features=self.features
            centroids=self.centroids
        else:
            ids=[self.f.index(feat) for feat in features]
            features=self.features[:,ids]
            centroids=self.centroids[:,ids]

        points=features.transpose()[0:3,:]
        labels=self.songs["Label"]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting the points, colored by cluster labels
        scatter = ax.scatter(points[0], points[1], points[2], c=labels, cmap='viridis',  s=50)
        scatterc=ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                            c='red', s=100, marker='x', alpha=1) 

        # Adding labels and title
        ax.set_xlabel(str(self.f[ids[0]]))
        ax.set_ylabel(str(self.f[ids[1]]))
        ax.set_zlabel(str(self.f[ids[2]]))
        ax.set_title('3D Scatter Plot of Songs')


        # Adding text to centroids
        for idx, xyz in enumerate(centroids):
            ax.text(xyz[0],xyz[1]+0.02,xyz[2]-0.05, str(idx), size='xx-large')
        return fig
# %%
def random_order(n,size=None):
    order=[]
    for subset in combinations([*range(n)],2):
        order+=list(subset)
    if size==None:
        order=order*max(1,int(24/n**2))
    else:
        times=int(np.ceil(size/len(order)))
        # print(size,'/',len(order),'=',times)

        # print(order,'*',times,'=',order*times)
        order=order*times
    return order

# %%
def split_genres(data):
    return [genre for genre in data['Genres'].str.split(',')]

# %%
def read_csv(filename,append=True,year=True,genres=True,timestamp=True):
    # read the data
    data = pandas.read_csv(filename)
    if append: 
        data["Track ID"]="https://open.spotify.com/track/"+data["Track ID"]
    if year:
        data["Release Year"]=data["Release Date"].str[:4].astype(int)
    if genres:
        data["Genres"]=split_genres(data)
    if timestamp:
        data["Added Timestamp"]=pandas.to_datetime(data["Added At"]).astype('int64') /10**9
    return data

# %%
def count_genres_in_labels(data,n,column,count_top=10):
    log_str=""
    counted_list=[]
    for i in range(n):
        g=[]
        data_filt=data.loc[data[column]==i]
        group_labelled=data_filt.dropna(subset="Genres")
        for genre in group_labelled["Genres"]:
            g=g+genre
        counter = Counter(g)
        c="\t".join([f"({c[0]}, {c[1]})".ljust(23) for c in counter.most_common(count_top)])
        log_str+=f"{i:<2} : {len(data_filt):<3} | {c}\n"
        counted_list.append(counter.most_common(count_top))
    return counted_list, log_str

# %%
def swap_items(data,elements, replacements):
    if len(elements) != len(replacements):
        raise ValueError("The 'elements' and 'replacements' lists must be of equal length.")

    # Create a bidirectional mapping
    swap_map = {}
    for orig, repl in zip(elements, replacements):
        swap_map[orig] = repl
        swap_map[repl] = orig

    # Replace items in the data list using the mapping.
    return [swap_map.get(item, item) for item in data]

# %%
def rotate_until(order,cluster):
    findWhere=np.where(order==cluster)
    try:
        shiftBy=np.min(findWhere)
    except ValueError:
        raise ValueError("Cluster is not found in order.")
    return order[shiftBy:]+order[:shiftBy]

# %%
def get_release_year(data):
    return [int(date[0:4]) if type(date)==str else date for date in data["Release Date"]]
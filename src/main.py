def main():
    import time
    import1=time.time()
    from playlist.manager import read_csv, count_genres_in_labels, Playlist, append_scalers
    print(f"Time to import manager: {time.time()-import1}")
    from pathlib import Path
    import2=time.time()
    from tagging.manual_tag import preprocess, process,make_names, sort_playlist
    print(f"Time to import tagging: {time.time()-import2}")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    from api.spotify import SpotifyManager
    import pandas
    from recommendation.recommender import sort_distances, sort_closest
    import numpy as np
    import os
    from dotenv import load_dotenv


    load_dotenv()
    SEED=40
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    REDIRECT_URI = os.getenv('REDIRECT_URI')  # This must match your Spotify Developer settings

    values=["Danceability","Energy","Valence","Acousticness","Speechiness","Instrumentalness","Tempo","Release Year",'Duration (ms)']
    n_clusters=50

    # Initialize

    # Spotify log in
    start_time=time.time()
    spotify = SpotifyManager(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
    spotify.login()
    print(f"Time to login: {time.time()-start_time}")

    # Read data (dld from exportify)
    current_dir= Path(__file__).parent
    data_path=current_dir.parent / "source" / "liked_songs_0224.csv"
    start_time=time.time()
    playlist=read_csv(data_path)
    time_read=time.time()
    print(f"Time to read: {time_read-start_time}")


    # Process data
    start_time=time.time()
    playlist,scaler=preprocess(playlist)
    labeled_data, scaler, kmeans=process(playlist,values,scaler,n_clusters,SEED)
    time_process=time.time()
    print(f"Time to process: {time_process-start_time}")
    labeled_data.to_csv("out/test_full_labeled.csv", sep=";",decimal=",",index=False)
    

    ## Loop through labels and push to Spotify
    # Genre stats
    start_time=time.time()
    label_to_genres,log=count_genres_in_labels(labeled_data,n_clusters,"Label 2",10000)
    time_count=time.time()
    print(f"Time to count: {time_count-start_time}")
    # Additional saving instead of print
    with open('log.txt', 'w', encoding='utf-8') as file:
        file.write(log)

    # Name the playlists to be saved
    # print(type(label_to_genres[0][0]))
    playlist_names=make_names(label_to_genres,3)
    # param=400 # cluster approx. size
    sorted_playlists=[]
    playlist_dict=[]    
    
    extra_values=["Added Timestamp"]
    sort_values=values+extra_values
    for i in range(n_clusters):
        # Process each labelled song
        playlist_cluster=labeled_data.loc[labeled_data["Label 2"]==i].copy().reset_index()
        # n=max(int(len(playlist_cluster)/param),2)
        if len(playlist_cluster)<2:
            print(f"{i} is <2.")
            continue
        mean=playlist_cluster[values].median(axis=0).to_frame().transpose()
        mean["Added Timestamp"]=max(playlist_cluster["Added Timestamp"])
        # index_latest=np.argmax(playlist_cluster["Added Timestamp"])
        # latest=playlist_cluster.loc[index_latest,:].to_frame().transpose()
        cluster_first=mean[sort_values]
        cluster_data=playlist_cluster[sort_values]
        ith_playlist=sort_closest(playlist_cluster,cluster_first,cluster_data)
        # ith_playlist=sort_playlist(playlist_cluster,n,values,-1)
        sorted_playlists.append(ith_playlist)

        # Create a playlist dict with song links
        name=f"{i:02}. {playlist_names[i]}"
        genres_sorted=[]
        for row in ith_playlist["Genres"]:
            if type(row)==list:
                for genre in row:
                    genres_sorted.append(genre)
        description=', '.join([l for l in pandas.unique(np.array(genres_sorted))])
        track_links = ith_playlist["Track ID"].to_list()
        if len(description)>300:
            description=description[0:description.rfind(', ',0,300)]

        playlist_dict.append({"name": name,"track_links": track_links, "description": description})
        print(f"Playlist {name} length: {len(track_links)}, description: {description}")
        print("========================================")

    # Write whole thing to file
    sorted_data=pandas.concat(sorted_playlists)
    sorted_data.to_csv("out/test_full_sorted.csv", sep=";",decimal=",",index=False)

    # Sync with Spotify 
    spotify.sync_playlists(playlist_dict)
    print(f"Playlists created: {len(playlist_dict)}")


if __name__=="__main__":
    main()


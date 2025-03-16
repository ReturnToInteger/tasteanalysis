def main():
    import time
    import1=time.time()
    from playlist.manager import read_csv, count_genres_in_labels
    print(f"Time to import manager: {time.time()-import1}")
    from pathlib import Path
    import2=time.time()
    from tagging.manual_tag import preprocess, process,make_names, sort_playlist, generate_sorted_playlists
    print(f"Time to import tagging: {time.time()-import2}")
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
    playlist=preprocess(playlist)
    labeled_data=process(playlist,values,n_clusters,SEED)
    time_process=time.time()
    print(f"Time to process: {time_process-start_time}")
    labeled_data.to_csv("out/test_full_labeled.csv", sep=";",decimal=",",index=False)
    

    ## Loop through labels and create playlist names, description and sorting
    # Genre stats
    start_time=time.time()
    label_to_genres,log=count_genres_in_labels(labeled_data,n_clusters,"Label 2",10000)
    time_count=time.time()
    print(f"Time to count: {time_count-start_time}")
    # Saving so you can see the stats for the latest version
    with open('log.txt', 'w', encoding='utf-8') as file:
        file.write(log)

    # Name the playlists to be saved
    playlist_names=make_names(label_to_genres,3)    
    
    extra_values=["Added Timestamp"]
    sort_values=values+extra_values

    sorted_playlists,playlist_dict=generate_sorted_playlists(labeled_data,playlist_names,n_clusters,sort_values,calc_first_helper)

    # # Write whole thing to file
    # sorted_data=pandas.concat(sorted_playlists)
    # sorted_data.to_csv("out/test_full_sorted.csv", sep=";",decimal=",",index=False)

    # # Sync with Spotify 
    # spotify.sync_playlists(playlist_dict)
    # print(f"Playlists created: {len(playlist_dict)}")

def calc_first_helper(data):
    mean=data.median(axis=0)
    mean["Added Timestamp"]=max(data["Added Timestamp"])
    # index_latest=np.argmax(data["Added Timestamp"])
    # latest=data.loc[index_latest,:]

    return mean

if __name__=="__main__":
    main()


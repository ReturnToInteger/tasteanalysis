# Playlist Clustering & Organizer

This is a personal learning project for analyzing and organizing music playlists. It uses clustering to group similar tracks and automatically creates and sorts Spotify playlists based on audio features and metadata.

Although it started as an experiment, it can be useful for anyone who wants to better organize their music library or generate themed playlists automatically.
## Features

 - 📊 Clusters songs using KMeans based on features like energy, valence, tempo, etc.
 - 🧼 Scales and processes audio features using MinMaxScaler.
 - 🧠 Groups tracks into playlists by musical similarity.
 - 🎧 Uses Spotify API to sync playlists directly to your Spotify account.
 - 🗂️ Automatically generates playlist names and descriptions based on genres.
 - 📁 Reads data from a Spotify export (CSV from Exportify).
 - 📝 Saves processed data and logs to file for later review.

## Project Structure

 - `playlist/` – functions for reading and organizing the dataset
 - `tagging/` – functions for naming and sorting playlists
 - `api/spotify.py` – handles Spotify login and playlist creation
 - `recommendation/` – custom logic for sorting tracks based on distance to cluster centers
 - `main()` – the script tying it all together
 - others - TO DO

## Requirements

Python 3.8+
Spotify Developer Account (to get CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
Dependencies: pandas, numpy, scikit-learn, python-dotenv, spotipy or similar

Install requirements with:

!TODO!
  
## Usage

 - Export your songs using Exportify.
 - Save the file as source/liked_songs_0224.csv or put your own file in root and modify the code.
 - Create a .env file in the root directory with the following:

```
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secre
REDIRECT_URI=http://localhost:your_redirect_port
```

 - Run the script:

`python main.py`

The script will:

 - Log in to Spotify
 - Read your liked songs
 - Cluster them into playlists
 - Sort each playlist
 - Sync the generated playlists to your Spotify account

## Notes

 - Playlist descriptions are built from genre tags.
 - You can tweak n_clusters, which features to include, or sorting strategy as you experiment.
 - The code includes timing logs to track performance.

## Why?

This project is mainly for learning about:

 - Clustering and recommendation systems
 - Working with real-world music data
 - Automating music organization
 - Using APIs in a structured way

But it could also help others make sense of large music libraries without manual curation.

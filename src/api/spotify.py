import spotipy
from spotipy.oauth2 import SpotifyOAuth
import re
import os
import json

class SpotifyManager:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.sp = None  # Will store the authenticated Spotify object
        self.user_id = None

    def login(self):
        """Authenticates the user and initializes Spotipy with the correct scope."""
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="playlist-modify-public playlist-modify-private"
        ))
        self.user_id = self.sp.me()["id"]
        print(f"Authenticated as: {self.user_id}")

    def extract_track_id(self, spotify_url):
        """Extracts the track ID from a Spotify track URL."""
        match = re.search(r"track/([a-zA-Z0-9]+)", spotify_url)
        return match.group(1) if match else None

    def create_playlist(self, playlist_name, track_links=None, description="Created via API"):
        """Creates a playlist and adds tracks in batches of 100."""
        if not self.sp:
            raise ValueError("User not authenticated. Call login() first.")

        playlist = self.sp.user_playlist_create(
            user=self.user_id, name=playlist_name, public=True, description=description
        )
        playlist_id = playlist["id"]
        print(f"Created Playlist: {playlist_name} (ID: {playlist_id})")

        if track_links:
            track_ids = [self.extract_track_id(link) for link in track_links if self.extract_track_id(link)]
            
            # Add tracks in chunks of 100
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i + 100]
                self.sp.playlist_add_items(playlist_id, batch)
                print(f"Added {len(batch)} tracks to playlist (Batch {i//100 + 1})")

        return playlist_id

    def modify_playlist(self, playlist_id, track_links, remove_existing=False, new_name=None, new_description=None):
        """
        Modifies an existing playlist.
        
        - If new_name or new_description are provided, updates the playlist details.
        - If remove_existing is True, replaces all tracks with the new ones.
        - Otherwise, appends the new tracks.
        Tracks are handled in batches of 100.
        """
        # Update playlist details if provided
        if new_name or new_description:
            self.sp.playlist_change_details(playlist_id, name=new_name, description=new_description)
            print("Updated playlist details.")

        # Process track links
        track_ids = [self.extract_track_id(link) for link in track_links if self.extract_track_id(link)]
        
        if remove_existing:
            # Replace playlist items with the first batch (up to 100 tracks)
            first_batch = track_ids[:100]
            self.sp.playlist_replace_items(playlist_id, first_batch)
            print(f"Replaced playlist items with {len(first_batch)} tracks.")
            # Append any remaining tracks, if there are more than 100
            if len(track_ids) > 100:
                for i in range(100, len(track_ids), 100):
                    batch = track_ids[i:i + 100]
                    self.sp.playlist_add_items(playlist_id, batch)
                    print(f"Appended {len(batch)} tracks to playlist (Batch {(i // 100) + 1}).")
            print("Playlist fully replaced with new tracks.")
        else:
            # Append new tracks in batches of 100
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i + 100]
                self.sp.playlist_add_items(playlist_id, batch)
                print(f"Appended {len(batch)} tracks to playlist (Batch {i // 100 + 1}).")
            print("New tracks appended to the existing playlist.")

    def sync_playlists(self, playlist_definitions, storage_file="playlists.json", default_remove_existing=True):
        """
        Synchronizes multiple playlists based on the provided definitions.
        
        playlist_definitions: A list of dictionaries where each dictionary can have:
            - "name": Playlist name (string, required)
            - "track_links": List of track URLs (list, required)
            - "description": Playlist description (string, optional)
            - "remove_existing": Boolean flag indicating if existing tracks should be replaced (optional)
        
        Behavior:
            - If there are already stored playlist IDs (from previous runs), these are reused.
            - If there are more stored playlists than definitions, the extra playlists are unfollowed (removed).
            - If there are fewer stored playlists than definitions, new playlists are created.
        
        The method updates the stored playlist IDs in the specified storage_file.
        """
        # Load existing playlist IDs, if available
        if os.path.exists(storage_file):
            with open(storage_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        new_playlist_ids = []

        # Process each playlist definition
        for i, p_def in enumerate(playlist_definitions):
            name = p_def["name"]
            track_links = p_def["track_links"]
            description = p_def.get("description", "Created via API")
            remove_existing = p_def.get("remove_existing", default_remove_existing)

            if i < len(existing_data):
                # Reuse an existing playlist ID and update it
                playlist_id = existing_data[i]["playlist_id"]
                print(f"Modifying existing playlist '{name}' (ID: {playlist_id})")
                self.modify_playlist(playlist_id, track_links, remove_existing, new_name=name, new_description=description)
            else:
                # Create a new playlist if no stored playlist exists
                playlist_id = self.create_playlist(name, track_links, description)
            new_playlist_ids.append({"playlist_id": playlist_id, "name": name})

        # If there are extra playlists stored that are no longer needed, remove (unfollow) them
        if len(existing_data) > len(playlist_definitions):
            for extra in existing_data[len(playlist_definitions):]:
                extra_id = extra["playlist_id"]
                self.sp.current_user_unfollow_playlist(extra_id)
                print(f"Removed extra playlist (ID: {extra_id}) from your library.")

        # Save the new list of playlist IDs to storage_file
        with open(storage_file, "w") as f:
            json.dump(new_playlist_ids, f, indent=4)
        print("Playlist sync complete.")
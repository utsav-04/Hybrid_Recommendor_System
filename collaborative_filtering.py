import os
import logging
import pandas as pd
import joblib
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, csr_matrix
from Data_cleaning import data_for_content_filtering

# Set up logging
logger = logging.getLogger('data_processing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

CLEANED_DATA_PATH = "data/cleaned_data.csv"
USER_HISTORY_PATH = "data/User Listening History.csv"
TRANSFORMED_DATA_SAVE_PATH = "data/transformed_data.npz"
TRACK_IDS_SAVE_PATH = "data/track_ids.npy"
INTERACTION_MATRIX_PATH = "data/interaction_matrix.npz"
FILTERED_DATA_PATH = "data/collab_filtered_data.csv"

# Columns to transform
frequency_encode_cols = ['year']
ohe_cols = ['artist', 'time_signature', 'key']
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]

def train_transformer(data: pd.DataFrame) -> None:
    try:
        transformer = ColumnTransformer(transformers=[
            ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
            ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
            ("standard_scale", StandardScaler(), standard_scale_cols),
            ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
        ], remainder='passthrough', n_jobs=-1, force_int_remainder_cols=False)
        transformer.fit(data)
        joblib.dump(transformer, "transformer.joblib")
        logger.debug("Transformer trained and saved successfully.")
    except Exception as e:
        logger.error("Error in training transformer: %s", e)
        raise

def transform_data(data: pd.DataFrame) -> np.ndarray:
    try:
        transformer = joblib.load("transformer.joblib")
        transformed_data = transformer.transform(data)
        logger.debug("Data transformation completed.")
        return transformed_data
    except Exception as e:
        logger.error("Error in transforming data: %s", e)
        raise

def save_transformed_data(transformed_data: np.ndarray, save_path: str) -> None:
    try:
        save_npz(save_path, transformed_data)
        logger.debug("Transformed data successfully saved at %s", save_path)
    except Exception as e:
        logger.error("Error saving transformed data: %s", e)
        raise

def calculate_similarity_scores(input_vector: np.ndarray, data: np.ndarray) -> np.ndarray:
    try:
        similarity_scores = cosine_similarity(input_vector, data)
        return similarity_scores
    except Exception as e:
        logger.error("Error in calculating similarity scores: %s", e)
        raise

def content_recommendation(song_name: str, artist_name: str, songs_data: pd.DataFrame, transformed_data: np.ndarray, k: int = 10) -> pd.DataFrame:
    try:
        song_name = song_name.lower()
        artist_name = artist_name.lower()
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        song_index = song_row.index[0]
        input_vector = transformed_data[song_index].reshape(1, -1)
        similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
        top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
        top_k_list = songs_data.iloc[top_k_songs_indexes][['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
        logger.debug("Generated content-based recommendations for %s by %s", song_name, artist_name)
        return top_k_list
    except Exception as e:
        logger.error("Error in generating content recommendations: %s", e)
        raise

def create_interaction_matrix(history_data: dd.DataFrame, track_ids_save_path: str, save_matrix_path: str) -> csr_matrix:
    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    df = df.categorize(columns=['user_id', 'track_id'])
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    track_ids = df['track_id'].cat.categories.values
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    df = df.assign(user_idx=user_mapping, track_idx=track_mapping)
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()
    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()
    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    save_npz(save_matrix_path, matrix)
    return matrix

def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)].sort_values(by="track_id").reset_index(drop=True)
    filtered_data.to_csv(save_df_path, index=False)
    return filtered_data

def collaborative_recommendation(song_name: str, artist_name: str, track_ids: list, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, k: int = 5) -> pd.DataFrame:
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    input_track_id = song_row['track_id'].values.item()
    ind = np.where(track_ids == input_track_id)[0].item()
    input_array = interaction_matrix[ind]
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_track_ids = track_ids[recommendation_indices]
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(), "score": top_scores})
    top_k_songs = (songs_data[songs_data["track_id"].isin(recommendation_track_ids)]
                   .merge(scores_df, on="track_id")
                   .sort_values(by="score", ascending=False)
                   .drop(columns=["track_id", "score"])
                   .reset_index(drop=True))
    return top_k_songs

def main(data_path: str, user_history_path: str) -> None:
    try:
        data = pd.read_csv(data_path)
        data_content = data_for_content_filtering(data)
        train_transformer(data_content)
        transformed_data = transform_data(data_content)
        save_transformed_data(transformed_data, TRANSFORMED_DATA_SAVE_PATH)

        user_data = dd.read_csv(user_history_path)
        unique_track_ids = user_data['track_id'].unique().compute().tolist()
        filtered_data = filter_songs_data(data, unique_track_ids, FILTERED_DATA_PATH)
        create_interaction_matrix(user_data, TRACK_IDS_SAVE_PATH, INTERACTION_MATRIX_PATH)

        logger.debug("Full pipeline (content + collaborative) executed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main(CLEANED_DATA_PATH, USER_HISTORY_PATH)

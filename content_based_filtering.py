import os
import logging
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
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

# Columns to transform
frequency_encode_cols = ['year']
ohe_cols = ['artist', 'time_signature', 'key']
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]

def train_transformer(data: pd.DataFrame) -> None:
    """Train and save a ColumnTransformer."""
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
    """Load transformer and transform data."""
    try:
        transformer = joblib.load("transformer.joblib")
        transformed_data = transformer.transform(data)
        logger.debug("Data transformation completed.")
        return transformed_data
    except Exception as e:
        logger.error("Error in transforming data: %s", e)
        raise

def save_transformed_data(transformed_data: np.ndarray, save_path: str) -> None:
    """Save transformed data to a file."""
    try:
        save_npz(save_path, transformed_data)
        logger.debug("Transformed data successfully saved at %s", save_path)
    except Exception as e:
        logger.error("Error saving transformed data: %s", e)
        raise

def calculate_similarity_scores(input_vector: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Calculate similarity scores using cosine similarity."""
    try:
        similarity_scores = cosine_similarity(input_vector, data)
        return similarity_scores
    except Exception as e:
        logger.error("Error in calculating similarity scores: %s", e)
        raise

def content_recommendation(song_name: str, artist_name: str, songs_data: pd.DataFrame, transformed_data: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Recommend top k similar songs based on content filtering."""
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

def main(data_path: str) -> None:
    """Main function to process data and generate transformer."""
    try:
        data = pd.read_csv(data_path)
        data_content_filtering = data_for_content_filtering(data)
        train_transformer(data_content_filtering)
        transformed_data = transform_data(data_content_filtering)
        save_transformed_data(transformed_data, "data/transformed_data.npz")
        logger.debug("Data processing and transformation pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main(CLEANED_DATA_PATH)

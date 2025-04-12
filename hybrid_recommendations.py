import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

# Set up logging
logger = logging.getLogger('recommendation_system')
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


class HybridRecommenderSystem:
    
    def __init__(self,  
                 number_of_recommendations: int, 
                 weight_content_based: float):
        
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based

    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        try:
            song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
            song_index = song_row.index[0]
            input_vector = transformed_matrix[song_index].reshape(1, -1)
            content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
            logger.debug("Content-based similarities calculated successfully.")
            return content_similarity_scores
        except Exception as e:
            logger.error("Error in calculating content-based similarities: %s", e)
            raise

    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        try:
            song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
            input_track_id = song_row['track_id'].values.item()
            ind = np.where(track_ids == input_track_id)[0].item()
            input_array = interaction_matrix[ind]
            collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
            logger.debug("Collaborative filtering similarities calculated successfully.")
            return collaborative_similarity_scores
        except Exception as e:
            logger.error("Error in calculating collaborative filtering similarities: %s", e)
            raise

    def __normalize_similarities(self, similarity_scores):
        try:
            minimum = np.min(similarity_scores)
            maximum = np.max(similarity_scores)
            normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
            logger.debug("Similarities normalized successfully.")
            return normalized_scores
        except Exception as e:
            logger.error("Error in normalizing similarities: %s", e)
            raise

    def __weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        try:
            weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
            logger.debug("Weighted combination of scores completed successfully.")
            return weighted_scores
        except Exception as e:
            logger.error("Error in combining scores: %s", e)
            raise

    def give_recommendations(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        try:
            # calculate content based similarities
            content_based_similarities = self.__calculate_content_based_similarities(
                song_name=song_name, artist_name=artist_name, songs_data=songs_data, transformed_matrix=transformed_matrix
            )

            # calculate collaborative filtering similarities
            collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(
                song_name=song_name, artist_name=artist_name, track_ids=track_ids, songs_data=songs_data, interaction_matrix=interaction_matrix
            )

            # normalize content based similarities
            normalized_content_based_similarities = self.__normalize_similarities(content_based_similarities)

            # normalize collaborative filtering similarities
            normalized_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)

            # weighted combination of similarities
            weighted_scores = self.__weighted_combination(
                content_based_scores=normalized_content_based_similarities,
                collaborative_filtering_scores=normalized_collaborative_filtering_similarities
            )

            # index values of recommendations
            recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]

            # get top k recommendations
            recommendation_track_ids = track_ids[recommendation_indices]

            # get top scores
            top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]

            # get the songs from data and print
            scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(), "score": top_scores})
            top_k_songs = (
                songs_data
                .loc[songs_data["track_id"].isin(recommendation_track_ids)]
                .merge(scores_df, on="track_id")
                .sort_values(by="score", ascending=False)
                .drop(columns=["track_id", "score"])
                .reset_index(drop=True)
            )

            logger.debug("Recommendations generated successfully.")
            return top_k_songs
        except Exception as e:
            logger.error("Error in generating recommendations: %s", e)
            raise

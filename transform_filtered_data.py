import pandas as pd
from Data_cleaning import data_for_content_filtering
from content_based_filtering import transform_data, save_transformed_data

# Path of filtered data
FILTERED_DATA_PATH = "data/collab_filtered_data.csv"

# Save path
TRANSFORMED_HYBRID_SAVE_PATH = "data/transformed_hybrid_data.npz"


def transform_filtered_data_for_hybrid(data_path: str, save_path: str) -> None:
    """
    This function loads filtered data, applies cleaning and transformation,
    and saves it for hybrid recommendation use.
    """
    try:
        # Load the filtered data
        filtered_data = pd.read_csv(data_path)

        # Clean the data
        filtered_data_cleaned = data_for_content_filtering(filtered_data)

        # Transform the data into a matrix
        transformed_data = transform_data(filtered_data_cleaned)

        # Save the transformed data
        save_transformed_data(transformed_data, save_path)

        print("Filtered data transformed and saved successfully.")

    except Exception as e:
        print(f"Error during hybrid transformation: {e}")


if __name__ == "__main__":
    transform_filtered_data_for_hybrid(FILTERED_DATA_PATH, TRANSFORMED_HYBRID_SAVE_PATH)

import os
import logging
import pandas as pd
import yaml

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


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while loading the data: %s', e)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the input DataFrame."""
    try:
        df = (
            df.drop_duplicates(subset='track_id')
              .drop(columns=['genre', 'spotify_id'])
              .fillna({'tags': 'no_tags'})
              .assign(
                  name=lambda x: x['name'].str.lower(),
                  artist=lambda x: x['artist'].str.lower(),
                  tags=lambda x: x['tags'].str.lower()
              )
              .reset_index(drop=True)
        )
        logger.debug('Data cleaning completed')
        return df
    except KeyError as e:
        logger.error('KeyError during data cleaning: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during data cleaning: %s', e)
        raise


def data_for_content_filtering(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.drop(columns=["track_id", "name", "spotify_preview_url"])
    )


def save_data(cleaned_data: pd.DataFrame, data_path: str) -> None:
    """Save cleaned dataset."""
    try:
        save_path = os.path.join(data_path, 'cleaned_data.csv')
        cleaned_data.to_csv(save_path, index=False)
        logger.debug('Cleaned data successfully saved in %s', save_path)
    except Exception as e:
        logger.error('An error occurred while saving data: %s', e)
        raise


def main():
    try:
        df = load_data(data_path='data/Music Info.csv')
        cleaned_df = clean_data(df)
        save_data(cleaned_df, data_path='data')
    except Exception as e:
        logger.error('Failed to complete the data cleaning and processing pipeline: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

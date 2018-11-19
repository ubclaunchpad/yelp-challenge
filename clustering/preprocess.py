import os

import numpy as np
import pandas as pd

BUSINESS_DATA = "data/yelp_academic_dataset_business.json"
CHECKIN_DATA = "data/yelp_academic_dataset_checkin.json"
REVIEW_DATA = "data/yelp_academic_dataset_review.json"

PROCESSED_DIR = "data/processed"
PROCESSED_RESTAURANTS = "restaurants.npy"


def filter_restaurants(businesses, max_restaurants=None):
    """
    Takes a DataFrame containing all types of Yelp businesses and returns
    one that only contains valid restaurants. If the `max_restaurants`
    argument is overrided, return only the N most popular restaurants
    (determined by looking at the review data).
    """

    # TODO: filter the input DataFrame
    return businesses


def create_feature_matrix(restaurants):
    """
    Returns a NumPy array that contains all of the features for
    restaurants that will be fed into the clustering model.
    """

    # TODO: form a valid NumPy array from the restaurant data
    return np.identity(1)


def preprocess(debug=False):
    with open(BUSINESS_DATA) as f:
        businesses = pd.read_json(f, lines=True)

    if debug:
        print(businesses)

    restaurants = filter_restaurants(businesses, max_restaurants=10000)
    matrix = create_feature_matrix(restaurants)

    # Serialize preprocessed matrix to disk
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    path = f'{PROCESSED_DIR}/{PROCESSED_RESTAURANTS}'
    np.save(path, matrix)
    print(f"Serialized restaurant data to {path}.")


if __name__ == '__main__':
    preprocess(debug=True)

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

    cols = ['business_id', 'is_open', 'review_count', 'categories']
    b = businesses[cols]

    review_limit = 10
    valid_categories = "Food|Restaurants"

    # Filters the businesses to find  open businesses with review counts higher than review_limit,
    # within categories specified in valid_categories.

    b = b.loc[b['review_count'] > review_limit]
    b = b.loc[b['is_open'] != 0]
    b = b[b['categories'].str.contains(valid_categories) == True]

    valid_businesses = b.sort_values(by=['review_count'], ascending=False)
    valid_businesses = valid_businesses.reset_index(drop=True)

    if max_restaurants:
        return valid_businesses[0:max_restaurants]
    else:
        return valid_businesses


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

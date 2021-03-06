import os
from collections import Counter

import numpy as np
import pandas as pd
import operator
import json

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

    b = businesses

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


def create_feature_matrix(res):
    """
    Returns a NumPy array that contains all of the features for
    restaurants that will be fed into the clustering model.
    """
    # Finds all unique restaurant categories
    categories = Counter()

    for index, row in res.iterrows():
        for category in row['categories'].split(', '):
            categories[category] += 1

    for index, cat in enumerate(categories.items()):
        # Only save categories that contain more than 8 restaurants
        if cat[1] < 8:
            continue

        res[cat[0]] = res.apply(lambda row: (1 if cat[0] in row['categories'] else 0), axis=1)

    # manually selected attributes that are relevant to restaurants and have a significant response rating according to kaggle
    # 'BusinessParking': "{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}"

    attributes = [
        'BusinessAcceptsCreditCards',
        'GoodForKids',
        'HasTV',
        'BikeParking',
        'WheelchairAccessible',
        'RestaurantsReservations',
        'HappyHour',
        'Caters',
        'DogsAllowed',
        'OutdoorSeating',
        'RestaurantsTakeOut',
        'RestaurantsDelivery'
    ]

    nonBoolAttributes = {
        'Alcohol': ['none', 'beer_and_wine', 'full_bar'],
        'RestaurantsPriceRange2':['2', '1', '3', '4'],
        'RestaurantsAttire': ['casual', 'dressy', 'formal'],
        'BusinessParking': ['garage', 'street', 'validated', 'lot', 'valet']
    }

    def addParkingColumns(row, parkingType):
        if row['attributes'] is not None and 'BusinessParking' in row['attributes'] and valType in row['attributes']['BusinessParking']:
            parking = json.loads(row['attributes']['BusinessParking'].replace('\'', '"').replace('False', '"False"').replace('True', '"True"'))
            if parking[parkingType] == 'True':
                return 1
            else:
                return 0
        else:
            return 0

    for att in attributes:
        res[att] = res.apply(lambda row: 1 if row['attributes'] is not None
                                          and att in row['attributes']
                                          and row['attributes'][att] == 'True'
                                          else 0, axis=1)

    # add non-boolean attributes
    for att in nonBoolAttributes:
        for valType in nonBoolAttributes[att]:
            if valType is 'none':
                continue
            key = att + '_' + valType;

            if att == 'BusinessParking':
                res[key] = res.apply(lambda row: addParkingColumns(row, valType), axis=1)
            else:
                res[key] = res.apply(lambda row:
                                              (1 if row['attributes'] is not None
                                               and att in row['attributes']
                                               and row['attributes'][att] == valType
                                               else 0), axis=1)
    # delete unnecessary columns
    deleteColumns = ['attributes', 'categories', 'hours', 'name', 'address', 'city', 'neighborhood', 'postal_code', 'state']

    res.drop(columns=deleteColumns)

    return res.values


def preprocess(debug=False):
    with open(BUSINESS_DATA) as f:
        businesses = pd.read_json(f, lines=True)

    restaurants = filter_restaurants(businesses, max_restaurants=10000)

    if debug:
        print(restaurants)

    matrix = create_feature_matrix(restaurants)

    print(matrix)

    # Serialize preprocessed matrix to disk
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    path = f'{PROCESSED_DIR}/{PROCESSED_RESTAURANTS}'
    np.save(path, matrix)
    print(f"Serialized restaurant data to {path}.")


if __name__ == '__main__':
    preprocess(debug=True)

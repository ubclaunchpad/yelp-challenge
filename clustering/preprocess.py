import pandas as pd

BUSINESS_DATA = "data/yelp_academic_dataset_business.json"
CHECKIN_DATA = "data/yelp_academic_dataset_checkin.json"
REVIEW_DATA = "data/yelp_academic_dataset_review.json"


def preprocess():
    with open(BUSINESS_DATA) as f:
        businesses = pd.read_json(f, lines=True)

    print(businesses)

    # TODO: Filter out non-businesses and serialize data back to disk.


if __name__ == '__main__':
    preprocess()

# yelp-challenge

UBC Launch Pad's entry for the Yelp Dataset Challenge


## Setup

Ensure that Python 3.6 is installed somewhere that `pipenv` can find it.
One way to do this is using [pyenv](https://github.com/pyenv/pyenv).

Download the [Yelp dataset](https://www.yelp.com/dataset/download)
and unzip it inside of `data/` subdirectory. Your directory structure should look like this:

```
$ tree data/
data
├── README.md
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_checkin.json
├── yelp_academic_dataset_photo.json
├── yelp_academic_dataset_review.json
├── yelp_academic_dataset_tip.json
└── yelp_academic_dataset_user.json
```


## Running

Running the clustering model preprocessing code:

```sh
pipenv install
pipenv run python clustering/preprocess.py
```

Opening a Jupyter notebook:

```sh
pipenv install
pipenv shell
jupyter notebook
```

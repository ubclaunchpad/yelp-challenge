import numpy as np
import re
import csv
import json
import os
import ast
import keras
from pathlib import Path
import pickle
from tqdm import tqdm

class Data(object):
    def __init__(self, data_source,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 n_classes=5):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.n_classes = n_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.max_len = 0
        self.data_source = data_source

        self.reviews = []
        self.ratings = []

    def load_data(self):
        print("Loading data...")
        all_review_data = open(self.data_source, encoding="utf-8")
        for line in all_review_data:
            curr_dict = ast.literal_eval(line)
            self.reviews.append(curr_dict["text"])
            self.ratings.append(curr_dict["stars"])

        self.max_len = len(max(self.reviews, key=len)) + 1
        print("Data loaded from " + self.data_source)
        all_review_data.close()

        return self.max_len, self.alphabet_size, len(self.ratings)

    def generate_all_data(self, save_reviews, save_ratings):
        print("Parsing reviews and ratings data...")
        reviews_path = Path(save_reviews)
        ratings_path = Path(save_ratings)
        if (reviews_path.is_dir() and ratings_path.is_file()):
            print("Reviews and ratings data already parsed and saved...These will be loaded now...")
            return

        os.makedirs(save_reviews)

        classes = dict()
        # one_hot = np.eye(self.n_classes, dtype='int64')
        for i, (s, c) in tqdm(enumerate(zip(self.reviews, self.ratings))):
            curr_review = np.asarray(self.str_to_indexes(s), dtype='int64')
            c = int(c) - 1
            np.save(os.path.join(reviews_path, str(i)), curr_review)
            classes[i] = c

        with open(save_ratings, 'wb') as fp:
            pickle.dump(classes, fp, protocol=-1)
        print("Reviews and ratings data successfully generated and saved...")

    def str_to_indexes(self, s):
        s = s.lower()
        max_length = min(len(s), self.max_len)
        str2idx = np.zeros(self.max_len, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, save_reviews, batch_size=32, dim=(1014,),
                 n_classes=5, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.save_reviews = save_reviews
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(self.save_reviews + '/' + str(ID) + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
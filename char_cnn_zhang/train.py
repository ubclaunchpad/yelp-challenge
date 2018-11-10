import json
import numpy as np
import pickle

from data_utils import Data, DataGenerator
from model import CharCNN
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))

    data_preprocessor = Data(data_source=config["data"]["data_source"],
                         alphabet=config["data"]["alphabet"],
                         n_classes=config["data"]["n_classes"])
    # input_size, alphabet_size, num_examples = data_preprocessor.load_data()
    input_size = 5058
    alphabet_size = 69
    # num_examples = 5996996
    num_examples = 10000
    data_preprocessor.generate_all_data(save_reviews=config["data"]["save_reviews"],
                                        save_ratings=config["data"]["save_ratings"])

    # Define training and validation splits
    partition = np.arange(num_examples)
    train_indices, valid_indices = train_test_split(partition,
                                                         test_size=0.05,
                                                         random_state=42,
                                                         shuffle=True)
    # Parameters
    params = {'dim': (input_size,),
              'batch_size': config["training"]["batch_size"],
              'n_classes': config["data"]["n_classes"],
              'shuffle': True}

    # Datasets
    with open(config["data"]["save_ratings"], 'rb') as fp:
        labels = pickle.load(fp)

    # Generators
    training_generator = DataGenerator(train_indices, labels, config["data"]["save_reviews"], **params)
    validation_generator = DataGenerator(valid_indices, labels, config["data"]["save_reviews"], **params)

    # Define model
    model = CharCNN(input_size=input_size,
                    alphabet_size=alphabet_size,
                     embedding_size=config["char_cnn"]["embedding_size"],
                     conv_layers=config["char_cnn"]["conv_layers"],
                     fully_connected_layers=config["char_cnn"]["fully_connected_layers"],
                     n_classes=config["data"]["n_classes"],
                     threshold=config["char_cnn"]["threshold"],
                     dropout_p=config["char_cnn"]["dropout_p"],
                     optimizer=config["char_cnn"]["optimizer"],
                     loss=config["char_cnn"]["loss"])
    # Train model
    model.train(training_gen=training_generator,
                validation_gen=validation_generator,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                log_freq=config["training"]["log_freq"])

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import multiprocessing as mp


class CharCNN(object):
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, n_classes,
                 threshold, dropout_p,
                 optimizer, loss):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.n_classes = n_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss

        print("Defining model...")
        self._build_model()

    def _build_model(self):
        # Input layer
        inputs = Input(shape=(self.input_size,), dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)

        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)

        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)

        # Output layer
        predictions = Dense(self.n_classes, activation='softmax')(x)

        # Build and compile model
        print("Compiling model...")
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=['accuracy']
                      )
        self.model = model
        self.model.summary()

    def train(self, training_gen, validation_gen,
              epochs, batch_size, log_freq=100):

        print("Crafting some useful callbacks...")
        # Create callbacks
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=log_freq, batch_size=batch_size,
                                  write_graph=False, write_grads=True, write_images=False,
                                  embeddings_freq=log_freq,
                                  embeddings_layer_names=None)
        checkpointer = ModelCheckpoint(filepath='char_cnn_best.h5',
                                       monitor='val_acc',
                                       verbose=2,
                                       save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_acc',
                                      patience=25,
                                      verbose=1)

        # Start training
        print("Training CharCNN model... ")
        history = self.model.fit_generator(generator=training_gen,
                                 epochs=epochs,
                                 verbose=1,
                                 # callbacks=[tensorboard, checkpointer, early_stopper],
                                 callbacks=[checkpointer, early_stopper],
                                 validation_data=validation_gen,
                                 workers=mp.cpu_count(),
                                 use_multiprocessing=True)

    def test(self, testing_inputs, testing_labels, batch_size):
        # Evaluate inputs
        self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
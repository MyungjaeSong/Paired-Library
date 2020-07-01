import argparse
import sys
from os.path import dirname
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from DeepCas9.utils import *

# BUFFER_SIZE = int(1e4)
# BATCH_SIZE = 500

sequence_column = 'Target context sequence (4+20+3+3)'
frequency_column = 'Background subtracted indel (%)'


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='DeepSpCas9',
                                     description='tensorflow2 implementation (similar to) DeepSpCas9 method')
    parser.add_argument('-training_file', default=dirname(__file__) + '/dataset/training.tsv')
    parser.add_argument('-testing_file', default=dirname(__file__) + '/dataset/endogenous_sites_test_set.tsv')
    parser.add_argument('-model_file', default=dirname(__file__) + '/model')
    parser.add_argument('--do_training', action='store_true', default=False,
                        help='train the model from training data (otherwise load trained model from -model_file)')
    parser.add_argument('-epochs', type=int, default=40)
    parser.add_argument('-validation_frac', type=float, default=0.1,
                        help='fraction of training set to leave for validation')
    parser.add_argument('-filter_size', type=int, default=7,
                        help='number of consecutive bp to learn local features from')
    parser.add_argument('-num_of_filters', type=int, default=50, help='number of filters in convolutional layer')
    parser.add_argument('-num_of_dense_layer_neurons', type=int, default=64, help='number of nuerons in dense layer')
    parser.add_argument('-dropout', type=float, default=0.2, help='dropout fraction')

    return parser.parse_args(argv[1:])


def build_model(filter_size, num_of_filters, num_of_dense_layer_neurons, dropout_rate):
    model = tf.keras.models.Sequential([
        # Apply filter each consecutive 7bp (7X4 matrix)
        keras.layers.Conv2D(num_of_filters, (filter_size, 4), activation='relu', input_shape=(30, 4, 1), padding='same',
                            kernel_regularizer=regularizers.l2(0.001)),
        # Downsamples the input representation by taking the maximum value over the window defined by pool_size
        # for each dimension along the features axis
        keras.layers.MaxPooling2D((2,1)),
        #keras.layers.Conv2D(50, (5, 4), activation='relu', padding='same',
        #                    kernel_regularizer=regularizers.l1(0.001)),
        #keras.layers.MaxPooling2D((2, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(num_of_dense_layer_neurons, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation='relu')
    ])

    model.summary()
    # compile loss function into model
    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=[keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def main():
    args = parse_args(sys.argv)
    training_file = args.training_file
    test_data_file = args.testing_file
    model_file = args.model_file
    do_training = args.do_training
    epochs = args.epochs
    validation_fraction = args.validation_frac
    filter_size = args.filter_size
    num_of_filters = args.num_of_filters
    num_of_dense_layer_neurons = args.num_of_dense_layer_neurons
    dropout_rate = args.dropout
    if do_training:
        trained_model = train_model(training_file, validation_fraction, filter_size, num_of_filters, num_of_dense_layer_neurons,
                                dropout_rate, epochs)
        trained_model.save(model_file)
    else:
        trained_model = keras.models.load_model(model_file)
    test_model(trained_model, test_data_file)


def train_model(training_file, validation_fraction, filter_size, num_of_filters, num_of_dense_layer_neurons, dropout, epochs):
    # Load data
    training_input_df = pd.read_csv(training_file, sep='\t')
    # Transform data (to tf compatible inputs)
    print(training_input_df[sequence_column])
    indel_frequencies = training_input_df[frequency_column].to_numpy()
    print("Convert sequence into one-hot encoding matrices")
    onehot_encoded_sequences_3d_matrix = series_to_onehot_encoding_3d_matrix(training_input_df[sequence_column])
    print(onehot_encoded_sequences_3d_matrix.shape)
    print(indel_frequencies.shape)
    # shuffle the data
    training_input_df.sample(frac=1)
    n_train_and_validation = training_input_df.shape[0]
    validation_set_size = int(n_train_and_validation * validation_fraction)
    # plt.hist(indel_frequencies, 30); plt.show()
    # Build the model
    model = build_model(filter_size, num_of_filters, num_of_dense_layer_neurons, dropout)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.002,
        decay_steps=10,
        decay_rate=0.8,
        staircase=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.1, patience=5, verbose=1, mode='auto',
        baseline=None, restore_best_weights=True
    )

    lr_scheduale_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # train the model
    history = model.fit(
        onehot_encoded_sequences_3d_matrix[validation_set_size:],
        indel_frequencies[validation_set_size:],
        epochs=epochs,
        validation_data=(onehot_encoded_sequences_3d_matrix[:validation_set_size],
                         indel_frequencies[:validation_set_size]),
        callbacks=[lr_scheduale_callback, early_stopping],
        verbose=1)
    # predict a subset of the training set (sanity)
    predicted = [x[0] for x in model.predict(onehot_encoded_sequences_3d_matrix[:validation_set_size])]
    actual = indel_frequencies[:validation_set_size]
    # Calculate pearson correlation
    predicted_arr = np.asarray(predicted)
    print(f'pearson correlation: {scipy.stats.pearsonr(predicted_arr, actual)}')

    visualize_history(history)
    return model


def test_model(trained_model, test_data_file):
    testing_input_df = pd.read_csv(test_data_file, sep='\t', skiprows=1, names=[sequence_column, frequency_column], usecols=[0, 1])
    one_hot_encoded_sequences_3d_matrix = series_to_onehot_encoding_3d_matrix(testing_input_df[sequence_column])
    actual_indel_frequency = testing_input_df[frequency_column]
    predicted_indel_frequencies_tensor = trained_model(one_hot_encoded_sequences_3d_matrix)
    predicted_indel_frequencies = tf.make_ndarray(tf.make_tensor_proto(predicted_indel_frequencies_tensor)).flatten()
    print(scipy.stats.pearsonr(predicted_indel_frequencies, actual_indel_frequency))
    plt.scatter(predicted_indel_frequencies, actual_indel_frequency)
    plt.show()




def series_to_onehot_encoding_3d_matrix(sequences_series):
    one_hot_encoded_sequences = sequences_series.apply(sequence_to_one_hot)
    onehot_encoded_sequences_3d_matrix = np.stack(one_hot_encoded_sequences.tolist())
    onehot_encoded_sequences_3d_matrix = np.expand_dims(onehot_encoded_sequences_3d_matrix, axis=3)
    return onehot_encoded_sequences_3d_matrix



if __name__ == "__main__":
    main()

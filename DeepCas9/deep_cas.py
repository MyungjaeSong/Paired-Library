from os.path import dirname

import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from tensorflow import keras



def sequence_to_one_hot(sequence):
    length = 30
    one_hot_encoded_sequence = np.zeros((length, 4), dtype=int)
    for i in range(length):
        if sequence[i] in "Aa":
            one_hot_encoded_sequence[i, 0] = 1
        elif sequence[i] in "Cc":
            one_hot_encoded_sequence[i, 1] = 1
        elif sequence[i] in "Gg":
            one_hot_encoded_sequence[i, 2] = 1
        elif sequence[i] in "Tt":
            one_hot_encoded_sequence[i, 3] = 1
        else:
            raise RuntimeError("Non-ATGC character " + sequence)
    return one_hot_encoded_sequence


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(30, 4)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    # compile loss function into model
    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

def main():

    training_file = dirname(__file__) + '/dataset/training.tsv'
    training_input_df = pd.read_csv(training_file, sep='\t')
    sequence_column = 'Target context sequence (4+20+3+3)'
    frequency_column = 'Background subtracted indel (%)'
    print(training_input_df[sequence_column])

    N_TRAIN_AND_VALIDATION = training_input_df.shape[0]
    N_VALIDATION = int(N_TRAIN_AND_VALIDATION / 10)
    N_TRAIN = N_TRAIN_AND_VALIDATION - N_VALIDATION
    EPOCHS = 20
    BUFFER_SIZE = int(1e4)

    print("Convert sequence into one-hot encoding matrices")
    onehot_encoded_sequences = training_input_df[sequence_column].apply(sequence_to_one_hot)
    onehot_encoded_sequences_3d_matrix = np.stack(onehot_encoded_sequences.tolist())

    indel_frequencies = training_input_df[frequency_column].to_numpy()
    print(onehot_encoded_sequences_3d_matrix.shape)
    print(indel_frequencies.shape)
    train_and_validation_dataset = tf.data.Dataset.from_tensor_slices((onehot_encoded_sequences_3d_matrix,  indel_frequencies))

    index = 0
    for x, y in train_and_validation_dataset:
        print(index)
        index += 1
        print(x)
        print(y)
        if index > 0:
            break

    train_and_validation_dataset.shuffle(BUFFER_SIZE)
    validate_ds = train_and_validation_dataset.take(N_VALIDATION)
    train_ds = train_and_validation_dataset.skip(N_VALIDATION).take(N_TRAIN).cache()

    # Build the model
    model = build_model()


    history = model.fit(
        onehot_encoded_sequences_3d_matrix,
        indel_frequencies,
        epochs=EPOCHS,
        validation_split=0.2, verbose=1)

    # predict a subset of the training set (sanity)
    predicted = [x[0] for x in model.predict(onehot_encoded_sequences_3d_matrix[-1000:])]
    actual = indel_frequencies[-1000:]

    # Calculate pearson correlation
    predicted_arr = np.asarray(predicted)
    print(f'pearson correlation: {scipy.stats.pearsonr(predicted_arr, actual)}')


if __name__ == "__main__":
    main()

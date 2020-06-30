import matplotlib.pyplot as plt
import numpy as np

def visualize_history(history):
    training_corr = history.history['correlation']
    validation_corr = history.history['val_correlation']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_corr) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_corr, 'r--')
    plt.plot(epoch_count, validation_corr, 'b-')
    plt.legend(['Training Corr', 'Validation Corr'])
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.show()


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
import matplotlib.pyplot as plt


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
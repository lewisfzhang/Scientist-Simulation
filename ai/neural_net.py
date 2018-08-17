import matplotlib as mpl

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import *
import subprocess as s


def main():
    clear_tb_board()

    EPOCHS = 100
    rms_k = 0.001

    use_sample = False
    if use_sample:
        boston_housing = keras.datasets.boston_housing

        (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data(
            path='/Users/stanford/Downloads/boston_housing.npz')
    else:
        big_data = np.load('../src/tmp/big_data.npy')
        # big_data = np.delete(big_data, (3, 5, 6), 1)  # remove all empty/not yet implemented columns
        print('Data set shape', big_data.shape)
        # data = x, labels = y
        # usage: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=#)
        len = big_data.shape[1] - 1 # get the number of parameters in the data set --> -1 = set index
        train_data, test_data, train_labels, test_labels = \
            train_test_split(big_data[:, :len], big_data[:, len], test_size=0.25)
        # print(train_data, '\n', test_data, '\n', train_labels, '\n', test_labels)

    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print("Training set: {}".format(train_data.shape))
    print("Testing set:  {}".format(test_data.shape))

    # Test data is *not* used when calculating the mean and std.
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = np.nan_to_num((train_data - mean) / std)  # converts all nan divide by 0s to 0
    test_data = np.nan_to_num((test_data - mean) / std)

    model = build_model(train_data, rms_k)

    model.summary()

    # all the callbacks are below

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tb_graphs', histogram_freq=0, write_graph=True, write_images=True)
    # base_logger = keras.callbacks.BaseLogger(stateful_metrics=None)

    # The patience parameter is the amount of epochs to check for improvement.
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
                        validation_split=0.2, verbose=2,
                        callbacks=[PrintDot(), tbCallBack])

    plot_history(history)

    [loss, mae, accuracy] = model.evaluate(test_data, test_labels, verbose=0)
    print("\n\nTesting set Mean Abs Error: ${:7.2f}".format(mae * 1000))
    print('Average loss:', loss)
    print('Average accuracy', accuracy)

    print('\nCheck to see if above data makes sense...')
    print('Predicted:', model.predict(test_data).flatten()[0])
    print('Actual:', test_labels[0])


def build_model(train_data, rms_k):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],), name='input_handler'),
        keras.layers.Dense(64, activation=tf.nn.relu, name='hidden1'),
        keras.layers.Dense(64, activation=tf.nn.relu, name='hidden2'),
        keras.layers.Dense(1, name='out')
    ])

    optimizer = tf.train.RMSPropOptimizer(rms_k)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def clear_tb_board():
    print('removing past log data for TF Boards...')
    path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
    s.call('rm -r '+path+'/ai/tb_graphs', shell=True)
    s.call('mkdir tb_graphs', shell=True)


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    # plt.show()


if __name__ == '__main__':
    main()

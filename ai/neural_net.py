# neural_net.py

import warnings as w
w.filterwarnings("ignore", message="numpy.dtype size changed")
w.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import *
import subprocess as s
import sklearn.metrics as metr
import pandas as pd
import os, sys
import src.config as config


def main():
    clear_tb_board()

    EPOCHS = 50
    rms_k = 0.001
    path = 'results/best_weights_{0}.{1}'
    scope = ''
    save_current = True

    use_sample = False
    if use_sample:
        boston_housing = keras.datasets.boston_housing

        (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data(
            path='/Users/stanford/Downloads/boston_housing.npz')
    else:
        big_data = np.load('../src/tmp/big_data{}.npy'.format(scope))
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

    train_model = True
    if train_model:
        # all the callbacks are below
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./tb_graphs', histogram_freq=0, write_graph=True, write_images=True)
        # monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=path.format(EPOCHS, 'hdf5'), verbose=0, save_best_only=True)  # save best model
        # base_logger = keras.callbacks.BaseLogger(stateful_metrics=None)

        # The patience parameter is the amount of epochs to check for improvement.
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        history = model.fit(train_data, train_labels, epochs=EPOCHS,
                            # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
                            validation_split=0.2, verbose=2,
                            callbacks=[PrintDot(), tbCallBack, checkpointer])

    switch = False
    if switch:
        plot_history(history)

        [loss, mae, accuracy] = model.evaluate(test_data, test_labels, verbose=0)
        print("\n\nTesting set Mean Abs Error: ${:7.2f}".format(mae))
        print('Average loss:', loss)
        print('Average accuracy', accuracy)

        print('\nCheck to see if above data makes sense...')
        print('Predicted:', model.predict(test_data).flatten()[0])
        print('Actual:', test_labels[0])
    else:
        model.load_weights(path.format(EPOCHS, 'hdf5'))  # load weights from best model

        # Predict and measure RMSE
        pred = model.predict(test_data)
        score = np.sqrt(metr.mean_squared_error(pred, test_labels))
        print("Score (RMSE): {}".format(score))

        # Plot the chart
        chart_regression(pred.flatten(), test_labels)  # sort is True by default
        chart_regression(pred.flatten(), test_labels, sort=False)

    print('\nsaving models...\n\n')
    if save_current:
        model.save('results/model.h5')
    model.save('results/model_{}.h5'.format(EPOCHS))


def build_model(train_data, rms_k):
    model = keras.Sequential([
        keras.layers.Dense(1000, activation=tf.nn.relu, input_shape=(train_data.shape[1],), name='input_handler'),
        keras.layers.Dense(500, activation=tf.nn.relu, name='hidden1'),
        keras.layers.Dense(250, activation=tf.nn.relu, name='hidden2'),
        keras.layers.Dense(1, name='out')
    ])

    # optimizer = tf.train.RMSPropOptimizer(rms_k)
    optimizer = tf.keras.optimizers.RMSprop(lr=rms_k)

    model.compile(loss='mse',
                  # optimizer='adam',
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    plt.plot(t['pred'].tolist(),label='prediction', color='orange')
    plt.plot(t['y'].tolist(),label='expected', color='blue')
    plt.ylabel('output')
    plt.legend()
    save_image('regression')


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
    save_image('history')


def save_image(name):
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    plt.savefig('plots/' + name)
    plt.close()


if __name__ == '__main__':
    # ensure current working directory is in src folder
    if os.getcwd()[-2:] != 'ai':
        # assuming we are somewhere inside the git directory
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        print('changing working directory from', os.getcwd(), 'to', path)
        os.chdir(path + '/ai')

    main()

import tensorflow as tf
import keras

import os

from datetime import datetime
from matplotlib import pyplot as plt


def train(model, train_ds, validation_ds, n_epochs, batch_size, with_plot=True, save_weights=False, checkpoint_path='savedModels/tf_ViT_covid_classifier', load_weights=False):

    n_epochs = n_epochs
    ds_train_batched = train_ds.batch(batch_size)

    accuracy_history, loss_history, validation_history = [], [], []

    loss_fn = tf.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    print('Begining training with %i Epochs' %n_epochs)
    for epoch in range(n_epochs):
        
        start_time = datetime.now()
        
        epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()
        epoch_loss = keras.metrics.Mean()
        validation_accuracy = keras.metrics.SparseCategoricalAccuracy()

        for x, y in ds_train_batched:
            with tf.GradientTape() as tape:
                y_ = model(x)
                loss = loss_fn(y, y_)
            gradient = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            epoch_accuracy.update_state(y, y_)
            epoch_loss.update_state(loss)
        
        #calculate validation accuracy
        for x, y in validation_ds.batch(16):
            y_ = model(x)
            validation_accuracy.update_state(y, y_)

        accuracy_history.append(epoch_accuracy.result())
        loss_history.append(epoch_loss.result())
        validation_history.append(validation_accuracy.result())
        # validation_history.append(validation_accuracy.result())
        
        end_time = datetime.now()
        elapsed = start_time - end_time
        print('Epoch: %i' %epoch, 'Validation Accuracy: %.4f' %validation_accuracy.result(), 'Epoch Accuracy: %.4f' %epoch_accuracy.result(), 
                'Loss: %.6f' %epoch_loss.result(), 'Elapsed Time: ' + elapsed.strftime("%H:%M:%S"))
        epoch_accuracy.reset_states()
        epoch_loss.reset_states()
        validation_accuracy.reset_states()

    if with_plot: plot(loss_history, accuracy_history)

    if save_weights: 
        model.save(checkpoint_path)
        print('Model Weights Saved')

def plot(loss_history, accuracy_history):
    plt.plot(loss_history, label='loss')
    plt.plot(accuracy_history, label='accuracy') 
    # plt.plot(validation_history, label='validation accuracy') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.show()

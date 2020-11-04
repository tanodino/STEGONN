import tensorflow as tf
import numpy as np
from utils.util import get_batch
from sklearn.metrics import accuracy_score


def train_step(model, x_train, y_train, optimizer, loss_object,  iterations=0, batch_size=32, x_validation=None,
               y_validation=None, train_neighborhoods=None, valid_neighborhoods=None, mask_train=None, mask_valid=None):
    loss_iteration = 0
    for ibatch in range(int(iterations)):
        batch_x = get_batch(x_train, ibatch, batch_size)
        batch_y = get_batch(y_train, ibatch, batch_size)
        batch_neighbors = get_batch(train_neighborhoods, ibatch, batch_size)
        mask_neighbors_batch = get_batch(mask_train, ibatch, batch_size)
        loss_iteration = update_gradients(model, batch_x, batch_y, batch_neighbors, mask_neighbors_batch, optimizer, loss_object, loss_iteration)
    total_loss = loss_iteration / int(iterations)
    return training_predictions(model, x_train, x_validation, train_neighborhoods, valid_neighborhoods, mask_train,
                                mask_valid, y_train, y_validation, total_loss, batch_size)


@tf.function
def update_gradients(model, batch_x, batch_y, batch_neighbors, mask_neighbors_batch, optimizer, loss_object, loss_iteration):
    with tf.GradientTape() as tape:
        predictions, predictions_aux, _, _ = model([batch_x, batch_neighbors, mask_neighbors_batch], training=True)
        loss_aux = loss_object(batch_y, predictions_aux)
        loss = loss_object(batch_y, predictions)
        tot_loss = loss + .5 * loss_aux
        loss_iteration += tot_loss
        gradients = tape.gradient(tot_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_iteration


"""
    makes training and validation sets predictions
"""


def training_predictions(model, x_train, x_validation, train_neighborhoods, valid_neighborhoods, mask_train, mask_valid,
                         y_train, y_validation, total_loss, batch_size):
    # compute metrics on training set after training step
    predictions, _ = model.predict_by_batch([x_train, train_neighborhoods, mask_train], batch_size=batch_size)
    train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(predictions, axis=1))

    # compute metrics on validation set
    predictions, validation_alphas = model.predict_by_batch([x_validation, valid_neighborhoods, mask_valid], batch_size=batch_size)
    valid_accuracy = accuracy_score(np.argmax(y_validation, axis=1), np.argmax(predictions, axis=1))
    return train_accuracy, valid_accuracy, validation_alphas, total_loss

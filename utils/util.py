import numpy as np
import tensorflow as tf
import time
import sys
from utils.constants import N_TIMESTAMPS, MAX_LIMIT, BATCH_SIZE
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score


def reshape_to_tensor(x):
    new_data = []
    for row in x:
        new_row = np.split(row, N_TIMESTAMPS)
        new_data.append(np.array(new_row))
    return np.array(new_data)


def get_batch(x, i, batch_size):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, x.shape[0])
    batch_x = x[start_id:end_id]
    return batch_x


def print_results(model_name, y_test, predictions):
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    f_score = f1_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), average='weighted')
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    f_score_class = f1_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), average=None)
    print("Accuracy score on test set:", accuracy)
    print("Kappa-score on test set: ", kappa)
    print("F-score on test set: ", f_score)
    print("Per class F-score on test set: ", f_score_class)


def print_epoch_metrics(train_accuracy, validation_accuracy, validation_alphas, total_loss, best_epoch_val_accuracy,
                        best_epoch_val_alphas, epoch, manager, ckpt, start):
    print("epoch %d loss %f Train Accuracy %f" % (epoch, total_loss, np.round(train_accuracy, 4)))
    if validation_accuracy >= best_epoch_val_accuracy:
        # new best model found, so save the checkpoint into a file
        best_epoch_val_accuracy = validation_accuracy
        best_epoch_val_alphas = validation_alphas
        best_step = int(ckpt.step)
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(best_step, save_path))
        print("Accuracy {:1.4f}".format(best_epoch_val_accuracy))
    print("vs. Validation Accuracy %f" % validation_accuracy)
    print("vs. BEST Validation Accuracy %f" % best_epoch_val_accuracy)
    end = time.time()
    print("epoch time: %f" % (end - start))
    print("===============")
    sys.stdout.flush()
    return best_epoch_val_accuracy, best_epoch_val_alphas


def shuffle_non_zero(x, dims):
    for i in range(len(x)):
        idx = np.arange(dims[i])
        perm = np.random.permutation(idx)
        x[i][idx] = x[i][perm]
    return x


def get_neighborhood_sizes(ids, rag):
    dims = []
    for id in ids:
        dims.append(len(list(rag.neighbors(id))))
    return np.array(dims)


def get_neighborhoods_and_mask(ids, rag, data, size=0):
    neighborhoods = []
    for id in ids:
        nx = [n for n in rag.neighbors(id)]
        neighbors = data[nx]
        neighborhoods.append(neighbors)
    padded_neighs = tf.keras.preprocessing.sequence.pad_sequences(neighborhoods, maxlen=size, padding='post', dtype='float32')
    reshaped_padded_neighs = []
    for p in range(len(padded_neighs)):
        reshaped_padded_neighs.append(reshape_to_tensor(padded_neighs[p]))
    reshaped_padded_neighs = np.array(reshaped_padded_neighs)
    mask = np.zeros((reshaped_padded_neighs.shape[0], size), dtype=np.bool)
    for i in range(len(neighborhoods)):
        mask[i, 0:len(neighborhoods[i])] = True
    return reshaped_padded_neighs, mask


def get_neighborhood_predictions(neighbors, pred):
    neighborhood_preds = []
    for n in neighbors:
        neighborhood_preds.append(pred[n])
    return neighborhood_preds


def test_step(model, manager, ckpt, x_test, y_test, test_neighborhoods, test_mask, test_dims):
    ckpt.restore(manager.latest_checkpoint)
    t_test_neighborhoods = shuffle_non_zero(test_neighborhoods, test_dims)
    t_test_neighborhoods = t_test_neighborhoods[:, 0:MAX_LIMIT, :, :]
    t_mask_test = test_mask[:, 0:MAX_LIMIT]
    pred, test_alphas, test_embeddings = model.predict_by_batch([x_test, t_test_neighborhoods, t_mask_test],
                                                                batch_size=BATCH_SIZE, return_embeddings=True)
    return pred, test_alphas





import numpy as np
from utils.constants import BASE_DIR_FOLDS, BASE_FILE_NAMES, EXTENSION
from utils.util import reshape_to_tensor


def load_dataset(current_fold):
    # make filenames using the current data fold number
    train_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[0] + current_fold + EXTENSION
    validation_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[1] + current_fold + EXTENSION
    test_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[2] + current_fold + EXTENSION

    target_train_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[3] + current_fold + EXTENSION
    target_validation_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[4] + current_fold + EXTENSION
    target_test_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[5] + current_fold + EXTENSION

    id_train_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[6] + current_fold + EXTENSION
    id_validation_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[7] + current_fold + EXTENSION
    id_test_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[8] + current_fold + EXTENSION

    # load of data
    x_train = np.load(train_fn).astype(np.float32)
    x_validation = np.load(validation_fn).astype(np.float32)
    x_test = np.load(test_fn).astype(np.float32)

    # reshaping of data into tensors of shape: (n_examples, n_timestamps, n_feature_bands)
    x_train = reshape_to_tensor(x_train)
    x_validation = reshape_to_tensor(x_validation)
    x_test = reshape_to_tensor(x_test)

    # load of labels
    y_train = np.load(target_train_fn)
    y_validation = np.load(target_validation_fn)
    y_test = np.load(target_test_fn)

    # load of segment ids
    id_train = np.load(id_train_fn)
    id_validation = np.load(id_validation_fn)
    id_test = np.load(id_test_fn)

    return x_train, x_validation, x_test, y_train, y_validation, y_test, id_train, id_validation, id_test

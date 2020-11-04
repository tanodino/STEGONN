import pickle
import numpy as np
import time
import tensorflow as tf
from sklearn.utils import shuffle
from models.starcane import STARCANE
from train.train import train_step
from utils.util import *
from utils.constants import *
from utils.data import load_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


global_data = np.load(GLOBAL_DATA_FILE_NAME).astype(np.float32)

# graph loading
max_neighborhood_size = 0
rag_file = open(RAG_FILE_NAME, 'rb')
rag = pickle.load(rag_file)
rag_file.close()
nodes = rag.nodes()
for id in nodes:
    max_neighborhood_size = max(len(list(rag.neighbors(id))), max_neighborhood_size)
fold = int(sys.argv[1])

print("Fold numero: ", fold)
x_train, x_validation, x_test, y_train, y_validation, y_test, id_train, id_validation, id_test = load_dataset(str(fold))

model = STARCANE(units=512, dropout_rate=0.4, n_classes=N_CLASSES)

train_dims = get_neighborhood_sizes(id_train, rag)
valid_dims = get_neighborhood_sizes(id_validation, rag)
test_dims = get_neighborhood_sizes(id_test, rag)

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, OUTPUT_FOLDER + "_" + str(fold), max_to_keep=1)

""" TRAINING """
iterations = x_train.shape[0] / BATCH_SIZE
if x_train.shape[0] % BATCH_SIZE != 0:
    iterations += 1

best_step = -1
best_epoch_val_accuracy = 0.0
best_epoch_val_alphas = []

train_neighborhoods, mask_train = get_neighborhoods_and_mask(id_train, rag, global_data, max_neighborhood_size)
valid_neighborhoods, mask_valid = get_neighborhoods_and_mask(id_validation, rag, global_data, max_neighborhood_size)
test_neighborhoods, mask_test = get_neighborhoods_and_mask(id_test, rag, global_data, max_neighborhood_size)
for e in range(EPOCHS):
    start = time.time()
    x_train, y_train, train_neighborhoods, train_dims, mask_train = shuffle(x_train, y_train, train_neighborhoods,
                                                                            train_dims, mask_train, random_state=0)
    train_neighborhoods = shuffle_non_zero(train_neighborhoods, train_dims)

    t_train_neighborhoods = train_neighborhoods[:, 0:MAX_LIMIT, :, :]
    t_valid_neighborhoods = valid_neighborhoods[:, 0:MAX_LIMIT, :, :]
    t_test_neighborhoods = test_neighborhoods[:, 0:MAX_LIMIT, :, :]

    t_mask_train = mask_train[:, 0:MAX_LIMIT]
    t_mask_valid = mask_valid[:, 0:MAX_LIMIT]
    t_mask_test = mask_test[:, 0:MAX_LIMIT]

    train_accuracy, validation_accuracy, validation_alphas, total_loss = train_step(model, x_train, y_train, optimizer, loss_object,
                                                                                    x_validation=x_validation, y_validation=y_validation,
                                                                                    train_neighborhoods=t_train_neighborhoods,
                                                                                    valid_neighborhoods=t_valid_neighborhoods,
                                                                                    mask_train=t_mask_train, mask_valid=t_mask_valid,
                                                                                    iterations=iterations, batch_size=BATCH_SIZE)
    best_epoch_val_accuracy, best_epoch_val_alphas = print_epoch_metrics(train_accuracy, validation_accuracy,
                                                                         validation_alphas, total_loss, best_epoch_val_accuracy,
                                                                         best_epoch_val_alphas, e, manager, ckpt, start)

""" TESTING """
pred, test_alphas = test_step(model, manager, ckpt, x_test, y_test, test_neighborhoods, mask_test, test_dims)
print_results("STARCANE with results: ", y_test, pred)
exit(0)

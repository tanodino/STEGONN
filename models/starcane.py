import tensorflow as tf
from layers.attention import AttentionLayerF, AttentionLayer
from layers.recurrent_cell import RecurrentCell
from models.cnn1d import Cnn
from utils.util import get_batch


"""
    STARCANE Model
"""


class STARCANE(tf.keras.Model):
    def __init__(self, units, n_classes, dropout_rate=0.0, hidden_activation='relu', output_activation='softmax',
                 name='Starcane', **kwargs):
        super(STARCANE, self).__init__(name=name, **kwargs)
        self.units = units
        self.f1 = Cnn(filters=self.units, kernel_size=3, dropout_rate=0.4)
        self.f2 = tf.keras.layers.RNN(RecurrentCell(self.units, dropout=dropout_rate), return_sequences=True, name="rnn_cell")
        self.attention = AttentionLayer(self.units)
        self.attention_f = AttentionLayerF(self.units, name="attention_embedding")
        self.hidden3 = tf.keras.layers.Dense(units=units, activation=hidden_activation, name="hidden_3")
        self.batchNorm3 = tf.keras.layers.BatchNormalization(name="batchnorm_hidden_3")
        self.hidden4 = tf.keras.layers.Dense(units=units, activation=hidden_activation, name="hidden_4")
        self.batchNorm4 = tf.keras.layers.BatchNormalization(name="batchnorm_hidden_4")
        self.clf_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation, name="output_model")
        self.clf_aux = tf.keras.layers.Dense(units=n_classes, activation=output_activation, name="output_model_aux")

    @tf.function
    def call(self, inputs, training=False):
        tgt_nodes = inputs[0]  # target nodes batch data shape: (batch_size, n_features)
        neighborhoods = inputs[1]  # neighborhoods batch data shape: (batch_size, max_neighborhood_size, n_features)
        mask = inputs[2]    # neighbors mask shape: (batch_size, max_neighborhood_size)
        mask = tf.dtypes.cast(mask, tf.float32)
        tgt_node_embedding = self.f1(tgt_nodes, training=training)  # (batch_size, 2*units)
        out_rnn = self.f2(neighborhoods, mask=mask, training=training)
        neigh_node_embedding, alphas = self.attention(inputs=[tgt_node_embedding, out_rnn], mask=mask, activation=tf.nn.leaky_relu)

        neighborhood_sizes = tf.expand_dims(tf.reduce_sum(mask, axis=1), axis=-1)   # shape: (batch_size, 1)
        neigh_node_embedding = neigh_node_embedding * neighborhood_sizes
        hidden_state = self.attention_f(inputs=[tgt_node_embedding, neigh_node_embedding])
        output = self.hidden3(hidden_state)
        output = self.batchNorm3(output, training=training)
        output = self.hidden4(output)
        output = self.batchNorm4(output, training=training)
        output_aux = self.clf_aux(hidden_state)
        return self.clf_output(output), output_aux, alphas, hidden_state

    def predict_by_batch(self, data, batch_size=32, return_embeddings=False):
        pred = None
        alphas = None
        embeddings = None
        x = data[0]
        neighbors = data[1]
        mask = data[2]
        iterations = x.shape[0] / batch_size
        if x.shape[0] % batch_size != 0:
            iterations += 1
        mask = tf.convert_to_tensor(mask, dtype=tf.bool)
        for ibatch in range(int(iterations)):
            batch_x = get_batch(x, ibatch, batch_size)
            batch_neighbors = get_batch(neighbors, ibatch, batch_size)
            mask_neighbors_batch = get_batch(mask, ibatch, batch_size)
            current_pred, _, current_alphas, current_h = self.call([batch_x, batch_neighbors, mask_neighbors_batch], training=False)
            if ibatch == 0:
                pred = current_pred
                alphas = current_alphas
                embeddings = current_h
            else:
                pred = tf.concat([pred, current_pred], axis=0)
                alphas = tf.concat([alphas, current_alphas], axis=0)
                embeddings = tf.concat([embeddings, current_h], axis=0)
        if return_embeddings:
            return pred, alphas, embeddings
        else:
            return pred, alphas

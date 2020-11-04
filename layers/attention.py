import tensorflow as tf


"""
    Attention Layers
"""


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, ch_output):
        super(AttentionLayer, self).__init__()
        self.ch_output = ch_output
        self.activation = tf.nn.leaky_relu
        self.output_activation = tf.keras.activations.softmax

    def build(self, input_shape):
        if len(input_shape) > 1:
            input_dim = input_shape[1]
        else:
            input_dim = input_shape
        self.A = self.add_weight(name="a_weight_matrix", shape=(2 * self.ch_output, 1))
        self.W = self.add_weight(name="W_target_nodes_weights", shape=[int(input_dim[-1]), self.ch_output])
        self.tgt_node_b = self.add_weight(name='bias_target', shape=(self.ch_output,), initializer='zeros')
        self.neigh_b = self.add_weight(name='bias_neigh', shape=(self.ch_output,), initializer='zeros')

    def call(self, inputs, **kwargs):
        hi = inputs[0]
        # target_nodes shape: batch_size x features_size F
        hj = inputs[1]
        # hj shape: batch_size x max(|N(x)|) x features_size F
        mask = tf.dtypes.cast(kwargs.get('mask'), tf.float32)
        # mask shape: batch_size x max(|N(x)|)

        whi = tf.nn.bias_add(tf.tensordot(hi, self.W, axes=1), self.tgt_node_b)
        # whi shape: batch_size x features_output F'
        whj = tf.nn.bias_add(tf.tensordot(hj, self.W, axes=1), self.neigh_b)
        # whj shape: batch_size x max(|N(x)|) x features_output F'
        multiply_dim = len(whj[0])
        whi = tf.tile(tf.expand_dims(whi, 1), multiples=(1, multiply_dim, 1))
        # whi shape for concat: batch_size x features_output F'
        concat = tf.concat([whi, whj], axis=2)
        # concat shape: batch_size x max(|N(x)|) x 2F'
        scores = self.activation(tf.tensordot(concat, self.A, axes=1))
        scores = tf.squeeze(scores, axis=-1)
        # scores shape: batch_size x max(|N(x)|)
        masked_scores = scores * mask
        alphas = self.output_activation(masked_scores)
        hj = hj * tf.expand_dims(alphas, -1)
        # hj shape: batch_size x max(|N(x)|) x features_output F'
        output = tf.reduce_sum(hj, axis=1)
        # output shape: (batch_size x features_output F')
        return output, alphas


class AttentionLayerF(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="att"):
        super(AttentionLayerF, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.activation = tf.nn.leaky_relu
        self.output_activation = tf.keras.activations.softmax

    def build(self, input_shape):
        if len(input_shape) > 1:
            input_dim = input_shape[1]
        else:
            input_dim = input_shape
        self.v = self.add_weight(name="a_weight_matrix", shape=(self.num_outputs, 1))
        self.W = self.add_weight(name="W_target_nodes_weights", shape=[int(input_dim[-1]), self.num_outputs])
        self.target_node_bias = self.add_weight(name='bias_target', shape=(self.num_outputs,), initializer='zeros')
        self.neighs_bias = self.add_weight(name='bias_neigh', shape=(self.num_outputs,), initializer='zeros')

    def call(self, inputs, **kwargs):
        tgt_node_embedding = inputs[0]
        # target_node_embedding shape: batch_size x features_size F
        neigh_embedding = inputs[1]
        # neighborhood_embedding shape: batch_size x max(|N(x)|) x features_size F

        whi = tf.nn.bias_add(tf.tensordot(tgt_node_embedding, self.W, axes=1), self.target_node_bias)
        # whi shape: batch_size x features_output F'
        whj = tf.nn.bias_add(tf.tensordot(neigh_embedding, self.W, axes=1), self.neighs_bias)
        # whj shape: batch_size x features_output F'

        whi = tf.matmul(tf.nn.tanh(whi), self.v)    # whi shape: batch_size x 1
        whj = tf.matmul(tf.nn.tanh(whj), self.v)    # whj shape: batch_size x 1
        scores = tf.concat([whi, whj], axis=-1)
        # scores shape: batch_size x 2
        alphas = self.output_activation(scores)
        alpha_t = tf.gather(alphas, [0], axis=1)
        alpha_n = tf.gather(alphas, [1], axis=1)
        tgt_node_embedding = tgt_node_embedding * alpha_t
        neigh_embedding = neigh_embedding * alpha_n
        return tgt_node_embedding + neigh_embedding

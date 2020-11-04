import tensorflow as tf
from models.cnn1d import Cnn


"""
    Custom Recurrent Cell
"""


class RecurrentCell(tf.keras.layers.Layer):
    def __init__(self, units=128, activation=None, recurrent_activation=None, use_bias=True, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, dropout=0., recurrent_dropout=0.):
        super(RecurrentCell, self).__init__()
        self.units = units
        self.output_size = self.units * 2
        self.state_size = self.units * 2
        self.recurrent_activation = recurrent_activation
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    def build(self, input_shape):
        self.cnn = Cnn(filters=self.units, kernel_size=3, dropout_rate=self.dropout)

    def call(self, inputs, states, training=False):
        h = self.cnn(inputs, training=training)
        return h, [h]
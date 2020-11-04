import tensorflow as tf

""""
    Convolutional Neural Network 1D Model
"""


class Cnn(tf.keras.Model):
    def __init__(self, filters=512, kernel_size=3, dropout_rate=0.4, activation='relu', name='CNN1D', **kwargs):
        super(Cnn, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=int(filters / 2), kernel_size=kernel_size, activation=activation)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv2 = tf.keras.layers.Conv1D(filters=int(filters / 2), kernel_size=kernel_size, activation=activation)
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv3 = tf.keras.layers.Conv1D(filters=int(filters / 2), kernel_size=kernel_size, activation=activation)
        self.batchNorm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv4 = tf.keras.layers.Conv1D(filters=int(filters / 2), kernel_size=kernel_size, activation=activation)
        self.batchNorm4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv5 = tf.keras.layers.Conv1D(filters=filters, strides=2, kernel_size=3, activation=activation)
        self.batchNorm5 = tf.keras.layers.BatchNormalization()
        self.dropout5 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv6 = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, activation=activation)
        self.batchNorm6 = tf.keras.layers.BatchNormalization()
        self.dropout6 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv7 = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation=activation)
        self.batchNorm7 = tf.keras.layers.BatchNormalization()
        self.dropout7 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv8 = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation=activation)
        self.batchNorm8 = tf.keras.layers.BatchNormalization()
        self.dropout8 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False):
        inputs = self.conv1(inputs)
        inputs = self.batchNorm1(inputs, training=training)
        inputs = self.dropout1(inputs, training=training)
        inputs = self.conv2(inputs)
        inputs = self.batchNorm2(inputs, training=training)
        inputs = self.dropout2(inputs, training=training)
        inputs = self.conv3(inputs)
        inputs = self.batchNorm3(inputs, training=training)
        inputs = self.dropout3(inputs, training=training)
        inputs = self.conv4(inputs)
        inputs = self.batchNorm4(inputs, training=training)
        inputs = self.dropout4(inputs, training=training)
        inputs = self.conv5(inputs)
        inputs = self.batchNorm5(inputs, training=training)
        inputs = self.dropout5(inputs, training=training)
        inputs = self.conv6(inputs)
        inputs = self.batchNorm6(inputs, training=training)
        inputs = self.dropout6(inputs, training=training)
        inputs = self.conv7(inputs)
        inputs = self.batchNorm7(inputs, training=training)
        conv7 = self.dropout7(inputs, training=training)
        conv8 = self.conv8(conv7)
        conv8 = self.batchNorm8(conv8, training=training)
        conv8 = self.dropout8(conv8, training=training)
        concat = tf.concat((conv8, conv7), axis=2)
        return self.pool(concat)
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input


class MyModel(tf.keras.Model):

    def __init__(self, input_layer):
        super().__init__()

        self.input_layer = input_layer

        self.g_conv1 = GraphLayer()
        self.dense1 = Dense(64)
        self.dropout1 = Dropout(0.5)

        self.g_conv2 = GraphLayer()
        self.dense2 = Dense(32)
        self.dropout2 = Dropout(0.5)

        self.out = Dense(4, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.input_layer(x)

        x = self.g_conv1([x, a])
        x = self.dense1(x)
        x = self.dropout2(x)

        x = self.g_conv2.call([x, a])
        x = self.dense2(x)
        x = self.dropout2(x)

        return self.out(x)


class GraphLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(GraphLayer, self).__init__()

        self.input_dim = input_dim
        self.units = units

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init,
                             shape=(input_dim, units),
                             dtype="float32",
                             trainable=True)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(1, input_shape[-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, x_input):
        feature_matrix, adjacency_matrix = x_input
        ax = tf.sparse.sparse_dense_matmul(
            tf.cast(adjacency_matrix, float),
            feature_matrix)

        z = ax * self.w

        return z


# model = Model()


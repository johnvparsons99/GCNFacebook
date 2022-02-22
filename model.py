import tensorflow
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Input, Layer


class MyModel(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__()

        self.g_conv1 = GraphLayer()
        self.g_conv2 = GraphLayer()

        self.dense1 = Dense(units=64)
        self.dense2 = Dense(units=32)

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

        self.out = Dense(units=4, activation='softmax')

    def call(self, inputs):
        x, a = inputs

        x = self.g_conv1([x, a])
        x = self.dense1(x)
        x = self.dropout2(x)

        x = self.g_conv2([x, a])
        x = self.dense2(x)
        x = self.dropout2(x)

        return self.out(x)


class GraphLayer(Layer):
    def __init__(self, units=32):
        super(GraphLayer, self).__init__()

        self.units = units
        self.w = []

    def build(self, input_shape):
        print("=============== BUILD ===============")
        print(input_shape)

        # noinspection INSPECTION_NAME
        self.w = self.add_weight(
            name="w",
            shape=(1, input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)
        print(self.w.shape)

    def call(self, x_input, **kwargs):
        feature_matrix, adjacency_matrix = x_input

        ax = tf.sparse.sparse_dense_matmul(
            tf.cast(adjacency_matrix, float),
            feature_matrix)

        z: tensorflow.Tensor = ax * self.w

        return z


# model = Model()


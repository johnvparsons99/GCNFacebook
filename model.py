import tensorflow as tf
from tensorflow import keras


class GCNLayer(tf.keras.layers.Layer):
    """
    My custom network layer.
    """
    def __init__(self, adj_m, test_adj_m):
        super(GCNLayer, self).__init__()
        self.adj_m = adj_m
        self.test_adj_m = test_adj_m
        self.w = tf.zeros_like(self.adj_m)

    def build(self, input_shape):
        self.w = self.add_weight("w",
                                       shape=(1, input_shape[-1]),
                                       initializer=keras.initializers.initializers_v1.RandomNormal)

    def call(self, feature_matrix, training=None, ):
        feature_matrix = tf.squeeze(feature_matrix)
        if training:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.adj_m, float), feature_matrix)
            z = ax * self.weights1
        else:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.test_adj_m, float), feature_matrix)
            z = ax * self.weights1

        return z


class Model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


model = Model()


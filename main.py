import numpy as np
import tensorflow as tf
import scipy
import sklearn
import matplotlib

import scipy.sparse as spr

import tensorflow.python.keras as keras
from tensorflow.python.keras import optimizers as ops
from tensorflow.python.keras.layers import Dense, Dropout, Input

import data_handler
import model
from model import MyModel

# ========================= GLOBAL VARIABLES =========================
# !!! IMPORTANT !!!
# Ensure valid file path to facebook.npz here
# Should be in the resources folder, in the same directory as this file
FILE_PATH = r"./resources/facebook.npz"

# Plotting Variables
PLOT_TSNE = False  # Set whether or not you want to plot accuracy
PLOT_ACCURACY = False  # Set whether or not you want to plot accuracy

# model Variables
EPOCHS = 200  # Set the number of epochs over which the Model should train
LEARNING_RATE = 0.01  # Set the Model learning rate

# Data Splitting Variables
TRAIN_SPLIT = 0.80
TEST_VAL_SPLIT = 0.20
# ====================================================================


def temp_masks(train_split, test_split, val_split, num_nodes):
    train_num = int(train_split * num_nodes)
    test_num = int(test_split * num_nodes)
    val_num = int(num_nodes - (train_num + test_num))

    train_base = tf.ones([train_num], tf.uint8)
    test_base = tf.ones([test_num], tf.uint8)
    val_base = tf.ones([val_num], tf.uint8)

    train_mask = tf.concat([train_base, tf.zeros_like(test_base), tf.zeros_like(val_base)], 0)
    test_mask = tf.concat([tf.zeros_like(train_base), test_base, tf.zeros_like(val_base)], 0)
    val_mask = tf.concat([tf.zeros_like(train_base), tf.zeros_like(test_base), val_base], 0)

    return train_mask, test_mask, val_mask


def main():
    print("=================================")
    print("======== STARTING PROGRAM =======")
    print("=================================")

    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)
    print("SciPy version:", scipy.__version__)
    print("SkLearn version:", sklearn.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    # print("Keras version:", keras.__version__)

    # Declare Variables
    # File Path
    filepath = 'facebook_large/facebook.npz'

    # Train/Test/Val/Split
    train = 0.80
    test = 0.10
    validate = 0.10

    # Parameters
    epochs = 100
    learning_rate = 0.001
    dropout_rate = 0.5

    # Parse Data
    data = np.load(filepath)
    features, edges, target = data_handler.parse_npz_files(data)

    # Split EdgeList into two tensors
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]

    # Build Adjacency Matrix
    # Convert EdgeList to Sparse Adjacency Matrix
    ones = tf.ones_like(page_one)  # Create Ones Matrix to set
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))  # Convert to SciPy COO Matrix
    a_bar.setdiag(1)  # Make all nodes adjacent to themselves
    a_bar = data_handler.normalize_adjacency_matrix(a_bar)
    a_bar = data_handler.coo_matrix_to_sparse_tensor(a_bar)
    a_bar = tf.sparse.reorder(a_bar)

    # Important Variables
    feats = tf.convert_to_tensor(data['features'])
    labels = tf.convert_to_tensor(data['target'])
    num_nodes = feats.shape[0]
    num_feats = feats.shape[1]

    # Split Train/Test/Validate
    train_mask, test_mask, val_mask = temp_masks(0.8, 0.1, 0.1, num_nodes)

    train_data = ([feats, a_bar], labels, train_mask)
    validation_data = ([feats, a_bar], labels, val_mask)
    test_data = ([feats, a_bar], labels, test_mask)

    # Inputs
    feats_input = keras.Input(shape=(num_feats,))
    a_bar_input = keras.Input(shape=(num_nodes,), sparse=True)

    # Model

    # Train Model
    gcn_model: keras.Model = MyModel(input_shape=tf.Tensor.get_shape(feats))

    gcn_model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    history = gcn_model.fit(x=(feats, a_bar),
                            y=labels,
                            sample_weight=train_mask,
                            batch_size=22470,
                            epochs=epochs,
                            shuffle=False,
                            validation_data=validation_data
                            )

    # Test Model
    # Evaluate Model
    # feats_test = feats[test_mask]
    # a_bar_test = a_bar
    # labels_test = labels[test_mask]

    gcn_model.evaluate(x=[feats, a_bar],
                       y=labels,
                       batch_size=22470,
                       sample_weight=test_mask,
                       )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


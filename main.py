import numpy as np
import tensorflow as tf
import keras
import scipy
import sklearn
import matplotlib

import scipy.sparse as spr

import data_handler


def temp_masks(train_split, test_split, val_split, num_nodes):
    train_num = int(train_split * num_nodes)
    test_num = int(test_split * num_nodes)
    val_num = int(num_nodes - (train_num + test_num))

    train_base = tf.ones([1, train_num], tf.uint8)
    test_base = tf.ones([1, test_num], tf.uint8)
    val_base = tf.ones([1, val_num], tf.uint8)

    train_mask = tf.concat([train_base, tf.zeros_like(test_base), tf.zeros_like(val_base)], 1)
    test_mask = tf.concat([tf.zeros_like(train_base), test_base, tf.zeros_like(val_base)], 1)
    val_mask = tf.concat([tf.zeros_like(train_base), tf.zeros_like(test_base), val_base], 1)

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
    print("Keras version:", keras.__version__)

    # Declare Variables
    filepath = 'facebook_large/facebook.npz'
    train = 80
    test = 10
    validate = 10

    # Parse Data
    data = np.load(filepath)
    features, edges, target = data_handler.parse_npz_files(data)


    # Split EdgeList into two tensors
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]
    # Features
    feats = tf.convert_to_tensor(data['features'])
    # Labels
    labels = tf.convert_to_tensor(data['target'])

    # Build Adjacency Matrix
    # Convert EdgeList to Sparse Adjacency Matrix
    ones = tf.ones_like(page_one)  # Create Ones Matrix to set
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))  # Convert to SciPy COO Matrix
    a_bar.setdiag(1)  # Make all nodes adjacent to themselves

    # Split Train/Test/Validate
    num_nodes = a_bar.shape[0]
    train_mask, test_mask, val_mask = temp_masks(0.8, 0.1, 0.1, 100)

    print(train_mask)
    print(test_mask)
    print(val_mask)

    # Train Model
    model = keras.Sequential

    # Test Model




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


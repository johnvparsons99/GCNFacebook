import json
import tensorflow as tf
import numpy as np
import sklearn
import scipy.sparse as spr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def coo_matrix_to_sparse_tensor(coo):
    """
        Converts a scipy COO Sparse Matrix to a Tensorflow Sparse Tensor
        Args:
            coo: Scipy COO Sparse Matrix to be converted

        Returns: The Tensorflow Sparse Tensor representation of the input matrix

        """
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def normalize_adjacency_matrix(a_bar):
    """
    Normalizes the Adjacency Matrix for the model by applying matrix
    multiplication between the Adjacency Matrix and the inverse square root
    of the D matrix.
    Args:
        a_bar: The Adjacency Matrix to normalise

    Returns: A normalised version of the input matrix

    """
    row_sum = np.array(a_bar.sum(1))
    d_inv_sqr = np.power(row_sum, -0.5).flatten()
    d_inv_sqr[np.isinf(d_inv_sqr)] = 0
    d_mat_inv_sqrt = spr.diags(d_inv_sqr)
    a_bar = a_bar.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return a_bar


def parse_npz_files(data):
    features = data['features']
    edges = data['edges']
    target = data['target']

    return features, edges, target


def summarise_data(data):
    print(data.files)
    print(data.shape)


def ensure_valid_split(train, test):
    """
    Checks that the combination of train and test is valid (i.e. if the sum to 1)
    Args:
        train: A float between 0-1, representing the portion of data to be used for training
        test: A float between 0-1, representing the portion of data to be used for testing/validation

    Returns: True if combination is valid, otherwise exits program.

    """
    if train+test == 1.0:
        return True
    else:
        print("Train Split + Validation Split + Test Split must equal 1.0.")
        print("Please ensure values for these variables sum to 1.0")
        exit(1)


def generate_masks(data_split, num_nodes):
    train_split, test_split, val_split = data_split
    train_num = int(train_split * num_nodes)
    test_num = int(test_split * num_nodes)
    val_num = int(num_nodes - (train_num + test_num))

    train_base = tf.ones([train_num], tf.uint8)
    test_base = tf.ones([test_num], tf.uint8)
    val_base = tf.ones([val_num], tf.uint8)

    train_mask = tf.concat([train_base, tf.zeros_like(test_base),
                            tf.zeros_like(val_base)], 0)
    test_mask = tf.concat([tf.zeros_like(train_base), test_base,
                           tf.zeros_like(val_base)], 0)
    val_mask = tf.concat([tf.zeros_like(train_base),
                          tf.zeros_like(test_base), val_base], 0)

    return train_mask, test_mask, val_mask


def parse_data(data):
    # Split EdgeList into two tensors
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]

    # Build Adjacency Matrix
    # Convert EdgeList to Sparse Adjacency Matrix
    ones = tf.ones_like(page_one)  # Create Ones Matrix to set
    a_bar = spr.coo_matrix(
        (ones, (page_one, page_two)))  # Convert to SciPy COO Matrix
    a_bar.setdiag(1)  # Make all nodes adjacent to themselves
    a_bar = normalize_adjacency_matrix(a_bar)
    a_bar = coo_matrix_to_sparse_tensor(a_bar)
    a_bar = tf.sparse.reorder(a_bar)

    # Important Variables
    feats = tf.convert_to_tensor(data['features'])
    labels = tf.convert_to_tensor(data['target'])
    num_nodes = feats.shape[0]
    num_feats = feats.shape[1]

    return a_bar, feats, labels, num_nodes


def generate_tsne_plot(labels, feats):
    """
    Generates and plots
    Args:
        labels:
        feats: The feature matrix of the nodes to be plotted

    Returns:

    """
    # TSNE
    print("Executing TSNE, this might take a moment...")
    tsne = TSNE(2)
    tsne_data = tsne.fit_transform(feats)

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_data[labels == 0, 0], tsne_data[labels == 0, 1], c='b',
                alpha=0.5, marker='.', label='TV Show')
    plt.scatter(tsne_data[labels == 1, 0], tsne_data[labels == 1, 1], c='r',
                alpha=0.5, marker='.', label='Company')
    plt.scatter(tsne_data[labels == 2, 0], tsne_data[labels == 2, 1], c='g',
                alpha=0.5, marker='.', label='Government')
    plt.scatter(tsne_data[labels == 3, 0], tsne_data[labels == 3, 1], c='m',
                alpha=0.5, marker='.', label='Politician')
    plt.title(f"GCN TSNE Plot Data")
    plt.legend()
    plt.show()


def generate_accuracy_plot(history):
    """
    Plots the Test Accuracy and Validation Accuracy of the model over the number of epochs.
    Args:
        history: history object generated by model.fit()

    """
    plt.plot(history.history['accuracy'], label='Test Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("GCN Accuracy over epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

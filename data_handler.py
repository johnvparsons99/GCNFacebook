import json
import tensorflow as tf
import numpy as np
import scipy.sparse as spr


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
    Normalizes the Adjacency Matrix for the model by applying matrix multiplication between the Adjacency Matrix and the
    inverse square root od the D matrix.
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




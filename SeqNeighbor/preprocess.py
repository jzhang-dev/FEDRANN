from scipy.sparse._csr import csr_matrix
import numpy as np


def tf_transform(feature_matrix: csr_matrix):
    row_sums = np.sum(feature_matrix, axis=1)
    feature_matrix /= row_sums.reshape(-1, 1)
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None):
    feature_matrix.data = np.ones_like(feature_matrix.data, dtype=np.float32)

    if idf is None:
        col_sums = feature_matrix.sum(axis=0).A1
        N = feature_matrix.shape[0]
        idf = np.log((N + 1) / (col_sums + 1)) + 1

    feature_matrix.data *= idf[feature_matrix.indices]

    return feature_matrix, idf

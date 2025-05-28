from scipy.sparse._csr import csr_matrix
import numpy as np


def tf_transform(feature_matrix: csr_matrix):
    row_sums = feature_matrix.sum(axis=1).A1
    feature_matrix /= row_sums
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None):
    feature_matrix.data = np.ones_like(feature_matrix.data, dtype=np.float32)

    if idf is None:
        col_sums = feature_matrix.sum(axis=0).A1
        assert feature_matrix.shape is not None
        N = feature_matrix.shape[0]
        idf = np.log((N + 1) / (col_sums + 1)) + 1

    feature_matrix.data *= idf[feature_matrix.indices]

    return feature_matrix, idf

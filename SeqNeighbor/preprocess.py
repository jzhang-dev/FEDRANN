
from scipy.sparse._csr import csr_matrix
import numpy as np

def manual_tf(data:csr_matrix):
    row_sums = np.sum(data, axis=1)
    tf_data = data / row_sums.reshape(-1, 1)
    tf_data = csr_matrix(tf_data) 
    return tf_data

def manual_idf(data:csr_matrix):
    binary_matrix = csr_matrix((np.ones_like(data.data), data.indices, data.indptr), shape=data.shape)
    col_sums = binary_matrix.sum(axis=0).A1
    N = binary_matrix.shape[0]
    idf = np.log((N + 1) / (col_sums + 1)) + 1 
    idf = idf.astype(binary_matrix.dtype)
    _data = binary_matrix.multiply(idf) 
    return _data,idf
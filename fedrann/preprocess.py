from scipy.sparse._csr import csr_matrix
from scipy.sparse import diags
import numpy as np
from .custom_logging import logger


def tf_transform(feature_matrix: csr_matrix):
    feature_matrix = feature_matrix.astype(np.float32)
    row_sums = feature_matrix.sum(axis=1).A1
    feature_matrix /= row_sums
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None):
    if idf is None:
        # Memory-efficient column sum
        logger.debug("Calculating column sum.")
        col_sums = np.asarray(feature_matrix.sum(axis=0)).ravel()
        assert feature_matrix.shape is not None
        nrow = feature_matrix.shape[0]
        
        logger.debug("Calculating IDF")
        idf = np.log(nrow / (col_sums.astype(np.float32) + 1e-12)).astype(np.float32)
    
    # Sparse matrix multiplication (memory-efficient)
    logger.debug("Applying IDF transformation")
    idf_diag = diags(idf, format='csc')
    feature_matrix.sort_indices()
    feature_matrix = feature_matrix.dot(idf_diag)
    
    return feature_matrix, idf

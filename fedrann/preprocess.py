from scipy.sparse._csr import csr_matrix
from scipy.sparse import diags
import numpy as np
from .custom_logging import logger


def tf_transform(feature_matrix: csr_matrix):
    feature_matrix = feature_matrix.astype(np.float32)
    row_sums = feature_matrix.sum(axis=1).A1
    feature_matrix /= row_sums
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None, *, chunk_size=int(1e9)):
    if idf is None:
        # Memory-efficient column sum
        logger.debug("Calculating column sum.")
        col_sums = np.asarray(feature_matrix.sum(axis=0)).ravel()
        assert feature_matrix.shape is not None
        nrow = feature_matrix.shape[0]
        
        logger.debug("Calculating IDF")
        idf = np.log(nrow / (col_sums.astype(np.float32) + 1e-12)).astype(np.float32)
    
    logger.debug("Applying IDF transformation")
    data = feature_matrix.data
    indices = feature_matrix.indices

    for i in range(0, len(data), chunk_size):
        end_idx = min(i+chunk_size, len(data))
        chunk = data[i:end_idx]
        data[i:end_idx] = chunk * idf[indices[i:end_idx]]

        progress = (end_idx / len(data))
        logger.debug(f"Progress: {progress:.2%}")
    
    return feature_matrix, idf

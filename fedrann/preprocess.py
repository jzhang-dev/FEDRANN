from scipy.sparse._csr import csr_matrix
from scipy.sparse import diags
import numpy as np
import sharedmem
from .custom_logging import logger


def tf_transform(feature_matrix: csr_matrix):
    feature_matrix = feature_matrix.astype(np.float32)
    row_sums = feature_matrix.sum(axis=1).A1
    feature_matrix /= row_sums
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None, *, threads: int = 1, chunk_size=int(100e6)):
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

    with sharedmem.MapReduce(np=threads) as pool:
        transformed_data = sharedmem.empty(len(data), dtype=np.float32) # type: ignore

        def work(i0):
            end_idx = min(i0 + chunk_size, len(data))
            chunk_data = data[i0:end_idx]
            chunk_indices = indices[i0:end_idx]
            transformed_chunk = chunk_data * idf[chunk_indices]
            with pool.critical:
                transformed_data[i0:end_idx] = transformed_chunk
            return i0
        
        def reduce(i0):
            end_idx = min(i0 + chunk_size, len(data))
            progress = (end_idx / len(data))
            logger.debug(f"Progress: {progress:.2%}")
            return i0
        
        pool.map(work, range(0, len(data), chunk_size), reduce=reduce)
    
    feature_matrix.data = transformed_data

    return feature_matrix, idf

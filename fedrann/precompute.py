from typing import cast
from dataclasses import dataclass
import math
import scipy as sp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as normalize_function
import subprocess
import tempfile
import os
from os.path import abspath, isfile, join
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, Optional, Iterable
import struct
from collections import namedtuple
from typing import Iterator

from .custom_logging import logger


def kmer_count_generator(filename: str,kmer_count: int):
    i = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('>'):
                count = int(line.strip()[1:])
            else: 
                yield i, count
                canonical_kmer_index = i + kmer_count
                yield canonical_kmer_index, count
                i +=1
                

def get_precompute_matrix(    
    n_components: int,
    counter_file: str,
    n_features: int,
    density: float | str = "auto",
    seed: int = 2094
):
    """
    读取C++生成的k-mer统计二进制文件
    返回: numpy数组，索引为kmer_index，值为count
    """
        

    result_array = np.zeros(n_features, dtype=np.uint64)
    
    # for kmer_index, count in kmer_count_generator(counter_file):
    for kmer_index, count in kmer_count_generator(counter_file,int(n_features/2)):
        result_array[kmer_index] = count
    
    idf = np.log((n_features) / (result_array + 1e-12)).astype(np.float32)

    if density == "auto":
        _density = 1 / math.sqrt(n_features)
    else:
        assert isinstance(density, float) and 0 < density <= 1
        _density = density
        
    rng = np.random.default_rng(seed)
    indices = []
    offset = 0
    indptr = [offset]
    for _ in range(n_components):
        # find the indices of the non-zero components for row i
        n_nonzero_i = rng.binomial(n_features, _density)
        indices_i = rng.choice(n_features, n_nonzero_i, replace=False)
        indices.append(indices_i)
        offset += n_nonzero_i
        indptr.append(offset)

    indices = np.concatenate(indices)

    # Among non zero components the probability of the sign is 50%/50%
    data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

    # build the CSR structure by concatenating the rows
    components = csr_matrix(
        (data, indices, indptr), shape=(n_components, n_features), dtype=np.float32
    )
    srp_matrix =  np.sqrt(1 / _density) / np.sqrt(n_components) * components

    idf = idf.reshape(-1, 1)
    logger.debug(f"{idf.shape=}")
    srp_matrix_t = srp_matrix.T
    logger.debug(f"{srp_matrix_t.shape=}")
    precompute_matrix = srp_matrix_t.multiply(idf)
    logger.debug(f"{precompute_matrix.shape=}")
    return precompute_matrix ,n_features






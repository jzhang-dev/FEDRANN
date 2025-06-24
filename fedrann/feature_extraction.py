import gzip, json, collections, gc, os
from typing import Sequence, Mapping, Collection, Optional, Iterable
from os.path import join
from multiprocessing import Pool, shared_memory
from functools import partial
from array import array
from Bio import SeqIO
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
from numpy.typing import NDArray
import xxhash
from numba import njit
import ahocorasick
import sharedmem



from .fastx_io import (
    FastqLoader,
    FastaLoader,
    get_reverse_complement_record,
    FastxRecord,
    reverse_complement,
    unzip,
    convert_fastq_to_fasta,
    make_fasta_index,
)
from .count_kmers import run_jellyfish, get_kmer_features


from . import globals
from .custom_logging import logger


def load_reads(
    file_path: str, batch_size: int
) -> Iterable[tuple[int, list[FastxRecord]]]:
    if (
        file_path.endswith(".fasta")
        or file_path.endswith(".fa")
        or file_path.endswith(".fasta.gz")
        or file_path.endswith(".fa.gz")
    ):
        # Load FASTA file
        loader = FastaLoader(file_path=file_path)
    elif (
        file_path.endswith(".fastq")
        or file_path.endswith(".fq")
        or file_path.endswith(".fastq.gz")
        or file_path.endswith(".fq.gz")
    ):
        loader = FastqLoader(file_path=file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a FASTA or FASTQ file."
        )
    i0 = 0
    batch_count = 0
    batch = []
    for record in loader:  # 迭代获取每条序列
        batch.append(record)
        batch.append(get_reverse_complement_record(record))
        if len(batch) >= batch_size:
            yield i0, batch
            i0 += len(batch)
            batch = []
            batch_count += 1
    yield i0, batch
    i0 += len(batch)
    logger.debug(f"Loaded {i0} records from {file_path} ({batch_count=} batches)")


@njit
def get_common_values(
    target: NDArray[np.uint64], query: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    """
    Numba优化的双指针法查找共有值

    参数:
        target: 已排序的一维numpy数组
        query: 已排序的一维numpy数组

    返回:
        包含两个数组共有值的numpy数组
    """
    # 初始化结果数组
    result = np.empty_like(query)
    count = 0

    t_ptr, q_ptr = 0, 0
    len_target, len_query = len(target), len(query)

    while t_ptr < len_target and q_ptr < len_query:
        t_val = target[t_ptr]
        q_val = query[q_ptr]

        if t_val == q_val:
            result[count] = t_val
            count += 1
            t_ptr += 1
            q_ptr += 1
        elif t_val < q_val:
            t_ptr += 1
        else:
            q_ptr += 1

    return result[:count]


def _get_matrix_density(matrix: csr_matrix) -> float:
    """
    Calculate the density of a sparse matrix.
    Density is defined as the ratio of non-zero elements to the total number of elements.
    """
    if not isinstance(matrix, csr_matrix):
        raise TypeError("Input must be a csr_matrix")
    assert matrix.shape is not None, "Matrix shape must be defined"
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    return matrix.nnz / (matrix.shape[0] * matrix.shape[1])


def get_hash_value(kmer: str, seed: int) -> int:
    return xxhash.xxh3_64(kmer, seed=seed).intdigest()




def get_feature_matrix(
    input_path: str,
    k: int,
    sample_fraction: float,
    min_multiplicity: int = 2,
):

    # Unzip
    if input_path.endswith(".gz"):
        input_unzipped_path = join(globals.temp_dir, os.path.basename(input_path[:-3]))
        unzip(input_path, input_unzipped_path)
    else:
        input_unzipped_path = input_path

    # Convert FASTQ to FASTA
    if input_unzipped_path.endswith(".fastq") or input_unzipped_path.endswith(".fq"):
        input_fasta_path = join(globals.temp_dir, "input.fasta")
        convert_fastq_to_fasta(input_unzipped_path, input_fasta_path, globals.threads)
    elif input_unzipped_path.endswith(".fasta") or input_unzipped_path.endswith(".fa"):
        input_fasta_path = input_unzipped_path
    else:
        raise ValueError(
            "Unsupported file format. Please provide a FASTA or FASTQ file."
        )

    data_array, col_indices_array, indptr_array, strands, read_names = get_kmer_features(
            input_fasta_path,
            k=k,
            sample_fraction=sample_fraction,
            min_multiplicity=min_multiplicity,
        )

    # Create sparse matrix
    logger.debug("Creating sparse feature matrix")
    logger.debug(f"{data_array.shape=}, {col_indices_array.shape=}, {indptr_array.shape=}")

    feature_matrix = csr_matrix(
        (data_array, col_indices_array, indptr_array),
        dtype=np.float32,
        copy=False,
    )

    logger.debug(
        f"{feature_matrix.shape=}, {len(feature_matrix.data)=}, density={_get_matrix_density(feature_matrix):.6f}"
    )

    return feature_matrix, read_names, strands


import gzip, json, collections, gc
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

from .fastx_io import (
    FastqLoader,
    FastaLoader,
    get_reverse_complement_record,
    FastxRecord,
    reverse_complement,
)
from .count_kmers import run_jellyfish


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


def _process_batch(batch, k: int, seed: int, shm_name: str, shm_shape) -> tuple:
    i0, records = batch
    if len(records) == 0:
        raise ValueError("Empty batch received")

    indices = array("L", [])
    hash_values = array("Q", [])
    multiplicity_values = array("L", [])

    shm = shared_memory.SharedMemory(name=shm_name)
    selected_hash_values = np.ndarray(shape=shm_shape, dtype=np.uint64, buffer=shm.buf)
    if len(selected_hash_values) == 0:
        raise ValueError("Zero selected k-mers provided")

    for i, record in enumerate(records):
        if len(record.sequence) < k:
            raise ValueError()
        sequence = record.sequence
        record_kmers = (sequence[p : p + k] for p in range(len(sequence) - k + 1))
        record_hash_values = [get_hash_value(kmer, seed=seed) for kmer in record_kmers]
        counts = collections.Counter(record_hash_values)
        unique_hashes = np.array(list(counts.keys()), dtype=np.uint64)
        unique_hashes.sort()
        common_hashes = get_common_values(selected_hash_values, unique_hashes)
        indices.extend([i0 + i] * len(common_hashes))
        hash_values.extend(common_hashes)
        multiplicity_values.extend(counts[h] for h in common_hashes)

    shm.close()
    read_names = [record.name for record in records]
    strands = [record.orientation for record in records]
    return i0, indices, hash_values, multiplicity_values, read_names, strands


def get_feature_matrix(
    input_path: str,
    k: int,
    sample_fraction: float,
    min_multiplicity: int = 2,
    batch_size: int = 10000,
):
    threads = globals.threads
    seed = globals.seed + 578

    logger.debug("Counting k-mers with Jellyfish")
    jellyfish_result = run_jellyfish(
        input_file=input_path,
        k=k,
        threads=threads,
        min_multiplicity=min_multiplicity,
        temp_dir=join(globals.output_dir, "jellyfish_temp"),
    )

    logger.debug("Loading reads")
    loader = load_reads(input_path, batch_size=batch_size)
    batches: dict[int, tuple[int, list[FastxRecord]] | None] = {
        i0: (i0, records) for i0, records in loader
    }

    logger.debug("Sampling k-mers")
    rng = np.random.default_rng(seed=seed + 23)
    selected_kmers = [
        kmer for kmer in jellyfish_result.get_result() if rng.random() < sample_fraction
    ]
    logger.debug(f"Sampled {len(selected_kmers)} k-mers")
    logger.debug("Computing reverse complements for selected k-mers")
    reverse_complement_kmers = [reverse_complement(kmer) for kmer in selected_kmers]
    selected_kmers.extend(reverse_complement_kmers)
    del reverse_complement_kmers
    logger.debug("Computing hash values for selected k-mers")
    selected_hash_values = np.array(
        [get_hash_value(kmer, seed=seed) for kmer in selected_kmers], dtype=np.uint64
    )
    del selected_kmers
    logger.debug("Sorting hash values")
    selected_hash_values.sort()

    logger.debug(f"Creating shared memory for selected hash values")
    shm = shared_memory.SharedMemory(create=True, size=selected_hash_values.nbytes)
    shm_array = np.ndarray(
        selected_hash_values.shape,
        dtype=selected_hash_values.dtype,
        buffer=shm.buf,
    )
    shm_shape = selected_hash_values.shape
    np.copyto(shm_array, selected_hash_values)
    del selected_hash_values
    gc.collect()

    logger.debug(f"Extracting k-mers from reads ({threads=} {batch_size=})")
    row_indices, col_indices, data = array("L", []), array("Q", []), array("L", [])
    read_names, strands = [], []
    fished_batches = 0

    def callback(
        i0,
        indices,
        hash_values,
        multiplicity_values,
        batch_read_names,
        batch_strands,
    ):
        nonlocal fished_batches

        batches[i0] = None
        row_indices.extend(indices)
        col_indices.extend(hash_values)
        data.extend(multiplicity_values)
        read_names.extend(batch_read_names)
        strands.extend(batch_strands)

        fished_batches += 1
        logger.debug("Progress: %.2f%%", fished_batches / len(batches) * 100)

    with Pool(threads, maxtasksperchild=100) as pool:
        work = partial(
            _process_batch, k=k, seed=seed, shm_name=shm.name, shm_shape=shm_shape
        )
        for result in pool.imap_unordered(work, batches.values()):
            callback(*result)
    shm.close()
    shm.unlink()

    if len(data) == 0:
        raise ValueError("No selected k-mers found in the input file.")
    if len(row_indices) != len(col_indices) or len(row_indices) != len(data):
        raise ValueError(
            f"{len(row_indices)=}, {len(col_indices)=}, {len(data)=} mismatch"
        )
    row_indices_numpy = np.array(row_indices, dtype=np.uint32)
    col_indices_numpy = np.array(col_indices, dtype=np.uint64)
    data_numpy = np.array(data, dtype=np.uint16)
    if len(row_indices_numpy) != len(col_indices_numpy) or len(
        row_indices_numpy
    ) != len(data_numpy):
        raise ValueError(
            f"{len(row_indices_numpy)=}, {len(col_indices_numpy)=}, {len(data_numpy)=} mismatch"
        )
    del row_indices, col_indices, data
    gc.collect()

    # # Filter out indices with multiplicity < min_multiplicity
    # logger.debug("Filtering k-mers by minimum multiplicity")
    # filtered_indices = _get_indices_by_min_count(
    #     col_indices_numpy, min_count=min_multiplicity
    # )
    # row_indices_numpy = row_indices_numpy[filtered_indices]
    # col_indices_numpy = col_indices_numpy[filtered_indices]
    # data_numpy = data_numpy[filtered_indices]
    # del filtered_indices
    # gc.collect()

    # Remove empty columns
    logger.debug("Removing empty columns")
    _, col_indices_numpy = np.unique(col_indices_numpy, return_inverse=True)
    if len(row_indices_numpy) != len(col_indices_numpy) or len(
        row_indices_numpy
    ) != len(data_numpy):
        raise ValueError(
            f"{len(row_indices_numpy)=}, {len(col_indices_numpy)=}, {len(data_numpy)=} mismatch"
        )

    # Create sparse matrix
    logger.debug("Creating sparse feature matrix")
    n_rows = row_indices_numpy.max() + 1
    n_cols = col_indices_numpy.max() + 1
    feature_matrix = csr_matrix(
        (data_numpy, (row_indices_numpy, col_indices_numpy)),
        shape=(n_rows, n_cols),
        dtype=np.uint32,
    )

    logger.debug(
        f"{feature_matrix.shape=}, {len(feature_matrix.data)=}, density={_get_matrix_density(feature_matrix):.6f}"
    )
    return feature_matrix, read_names, strands

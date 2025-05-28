import gzip, json, collections, gc
from typing import Sequence, Mapping, Collection, Optional, Iterable
from array import array
from Bio import SeqIO
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from .fastx_io import (
    FastqLoader,
    FastaLoader,
    get_reverse_complement_record,
    FastxRecord,
)
import time
import xxhash
from multiprocessing import Pool
from functools import partial
import gc
import sharedmem

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
    batch = []
    for record in loader:  # 迭代获取每条序列
        batch.append(record)
        batch.append(get_reverse_complement_record(record))
        if len(batch) >= batch_size:
            yield i0, batch
            i0 += len(batch)
            batch = []


def _get_indices_by_min_count(x: NDArray, min_count: int = 2) -> NDArray:
    uniq_vals, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)

    # 找出出现次数>=2的值的索引（在uniq_vals中的索引）
    mask = counts >= min_count
    target_value_indices = np.where(mask)[0]  # 例如 [2, 3] 对应值为 3 和 4

    # 在原数组中查找这些值的索引
    matching_indices = np.isin(inverse, target_value_indices).nonzero()[0]
    return matching_indices


def _remove_empty_columns(x: csr_matrix) -> csr_matrix:
    # 计算每列的非零元素数
    col_counts = x.getnnz(axis=0)
    # 获取非空列的索引
    non_empty_cols = np.where(col_counts > 0)[0]
    # 保留非空列
    return x[:, non_empty_cols]


def get_feature_matrix(
    input_path: str,
    k: int,
    sample_fraction: float,
    min_multiplicity: int = 2,
    batch_size: int = 10_000,
):
    threads = globals.threads
    seed = globals.seed + 578

    max_hash = int(2**64 - 1 * sample_fraction)
    loader = load_reads(input_path, batch_size=batch_size)

    with sharedmem.MapReduce(np=threads) as pool:

        def work(batch):
            i0, records = batch
            indices = array("L", [])
            hash_values = array("Q", [])
            multiplicity_values = array("H", [])
            max_multiplicity = 2**16 - 1

            for i, record in enumerate(records):
                if len(record.sequence) < k:
                    raise ValueError()

                counts = collections.Counter()
                kmers = (
                    record.sequence[p : p + k]
                    for p in range(len(record.sequence) - k + 1)
                )
                for kmer in kmers:
                    hash_value = xxhash.xxh3_64(kmer, seed=seed).intdigest()
                    if hash_value < max_hash:
                        counts[hash_value] += 1

                counts = {
                    hash_value: min(multiplicity, max_multiplicity)
                    for hash_value, multiplicity in counts.items()
                }
                indices.extend([i0 + i] * len(counts))
                hash_values.extend(counts.keys())
                multiplicity_values.extend(counts.values())

            read_names = [record.name for record in records]
            strands = [record.orientation for record in records]
            return i0, indices, hash_values, multiplicity_values, read_names, strands

        logger.debug("Loading reads")
        batches: dict[int, tuple[int, list[FastxRecord]] | None] = {
            i0: (i0, records) for i0, records in loader
        }

        row_indices, col_indices, data = array("L", []), array("Q", []), array("H", [])
        read_names, strands = [], []

        def reduce(
            i0,
            indices,
            hash_values,
            multiplicity_values,
            batch_read_names,
            batch_strands,
        ):
            batches[i0] = None
            gc.collect()

            row_indices.extend(indices)
            col_indices.extend(hash_values)
            data.extend(multiplicity_values)
            read_names.extend(batch_read_names)
            strands.extend(batch_strands)

        logger.debug(f"Extracting k-mers from reads ({threads=} {batch_size=})")
        pool.map(work, batches.values(), reduce=reduce)

    row_indices_numpy = np.frombuffer(row_indices, dtype=np.uint32)
    col_indices_numpy = np.frombuffer(col_indices, dtype=np.uint64)
    data_numpy = np.frombuffer(data, dtype=np.uint16)
    del row_indices, col_indices, data
    gc.collect()

    # Filter out indices with multiplicity < min_multiplicity
    logger.debug("Filtering k-mers by minimum multiplicity")
    filtered_indices = _get_indices_by_min_count(
        col_indices_numpy, min_count=min_multiplicity
    )
    row_indices_numpy = row_indices_numpy[filtered_indices]
    col_indices_numpy = col_indices_numpy[filtered_indices]
    data_numpy = data_numpy[filtered_indices]
    del filtered_indices
    gc.collect()

    # Remove empty columns
    logger.debug("Removing empty columns")
    _, col_indices_numpy = np.unique(col_indices_numpy, return_inverse=True)

    # Create sparse matrix
    logger.debug("Creating sparse feature matrix")
    n_rows = row_indices_numpy.max() + 1
    n_cols = col_indices_numpy.max() + 1
    feature_matrix = csr_matrix(
        (data_numpy, (row_indices_numpy, col_indices_numpy)),
        shape=(n_rows, n_cols),
        dtype=np.uint16,
    )

    logger.debug("Feature matrix shape: %s", feature_matrix.shape)
    return feature_matrix, read_names, strands

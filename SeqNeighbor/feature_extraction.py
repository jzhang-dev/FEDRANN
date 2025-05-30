import gzip, json, collections, gc
from typing import Sequence, Mapping, Collection, Optional, Iterable
from os.path import join
from multiprocessing import Pool, set_start_method
from functools import partial
from array import array
from Bio import SeqIO
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
from numpy.typing import NDArray
import xxhash

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
    batch = []
    for record in loader:  # 迭代获取每条序列
        batch.append(record)
        batch.append(get_reverse_complement_record(record))
        if len(batch) >= batch_size:
            yield i0, batch
            i0 += len(batch)
            batch = []
    yield i0, batch
    i0 += len(batch)
    logger.debug("Total records loaded: %d", i0)


def _get_indices_by_min_count(x: NDArray, min_count: int = 2) -> NDArray:
    uniq_vals, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)

    # 找出出现次数>=2的值的索引（在uniq_vals中的索引）
    mask = counts >= min_count
    target_value_indices = np.where(mask)[0]  # 例如 [2, 3] 对应值为 3 和 4

    # 在原数组中查找这些值的索引
    matching_indices = np.isin(inverse, target_value_indices).nonzero()[0]
    return matching_indices

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


def _process_batch(
    batch, k: int, seed: int, selected_hash_values: set[int]
) -> tuple:
    i0, records = batch
    if len(records) == 0:
        raise ValueError("Empty batch received")

    indices = array("L", [])
    hash_values = array("Q", [])
    multiplicity_values = array("L", [])

    if len(selected_hash_values) == 0:
        raise ValueError("Zero selected k-mers provided")

    for i, record in enumerate(records):
        if len(record.sequence) < k:
            raise ValueError()

        counts = collections.Counter()
        sequence = record.sequence
        kmers = (sequence[p : p + k] for p in range(len(sequence) - k + 1))
        for kmer in kmers:
            hash_value = get_hash_value(kmer, seed=seed)
            if hash_value in selected_hash_values:
                counts[hash_value] += 1

        indices.extend([i0 + i] * len(counts))
        hash_values.extend(counts.keys())
        multiplicity_values.extend(counts.values())

    read_names = [record.name for record in records]
    strands = [record.orientation for record in records]
    return i0, indices, hash_values, multiplicity_values, read_names, strands


def get_feature_matrix(
    input_path: str,
    k: int,
    sample_fraction: float,
    min_multiplicity: int = 2,
    batch_size: int = 50_000,
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
        kmer
        for kmer in jellyfish_result.get_result()
        if rng.random() < sample_fraction
    ]
    logger.debug(f"Sampled {len(selected_kmers)} k-mers")
    logger.debug("Computing reverse complements for selected k-mers")
    reverse_complement_kmers = [reverse_complement(kmer) for kmer in selected_kmers]
    selected_kmers.extend(reverse_complement_kmers)
    del reverse_complement_kmers
    logger.debug("Computing hash values for selected k-mers")
    selected_hash_values: set[int] = set(get_hash_value(kmer, seed=seed) for kmer in selected_kmers)
    del selected_kmers

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

    logger.debug(f"Extracting k-mers from reads ({threads=} {batch_size=})")
    with Pool(threads, maxtasksperchild=100) as pool:
        work = partial(
            _process_batch, k=k, seed=seed, selected_hash_values=selected_hash_values
        )
        for result in pool.imap_unordered(work, batches.values()):
            callback(*result)

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

    logger.debug(f"{feature_matrix.shape=}, {len(feature_matrix.data)=}, density={_get_matrix_density(feature_matrix):.6f}")
    return feature_matrix, read_names, strands

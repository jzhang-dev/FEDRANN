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




def get_feature_matrix_1(
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
        convert_fastq_to_fasta(input_unzipped_path, input_fasta_path)
    elif input_unzipped_path.endswith(".fasta") or input_unzipped_path.endswith(".fa"):
        input_fasta_path = input_unzipped_path
    else:
        raise ValueError(
            "Unsupported file format. Please provide a FASTA or FASTQ file."
        )

    # Get features
    col_indices = array("Q", [])  # uint64
    indptr = array("Q", [0])  # uint64
    read_names = []
    strands = []


    for i, (name, indices, strand) in enumerate(
        get_kmer_features(
            input_fasta_path,
            k=k,
            sample_fraction=sample_fraction,
            min_multiplicity=min_multiplicity,
        )
    ):
        col_indices.extend(indices)
        indptr.append(len(col_indices))
        read_names.append(name)
        strands.append(strand)

        if (i+1) % 100000 == 0:
            logger.debug(f"Processed {i+1} records: {len(col_indices)=}")

    # Create sparse matrix
    logger.debug("Creating sparse feature matrix")
    col_indices_array = np.frombuffer(col_indices, dtype=np.uint64)
    data_array = np.ones_like(col_indices_array, dtype=np.float32)
    indptr_array = np.frombuffer(indptr, dtype=np.uint64)
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


def _build_automaton(kmers: Sequence[str]) -> ahocorasick.Automaton:
    # Function to build the Aho-Corasick automaton
    A = ahocorasick.Automaton()
    for i, kmer in enumerate(kmers):
        A.add_word(kmer, i)
    A.make_automaton()
    return A


def get_feature_matrix_2(
    input_path: str,
    k: int,
    sample_fraction: float,
    min_multiplicity: int = 2,
    batch_size: int = 10000,
):
    # Aho-Corasick
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

    logger.debug("Building Aho-Corasick automaton")
    automaton = _build_automaton(selected_kmers)

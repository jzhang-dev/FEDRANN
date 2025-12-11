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
import struct

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


def parse_kmer_searcher_output(ks_file: str, kmer_count):
    with open(ks_file, "rb", buffering=1024 ** 3) as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError("文件头不完整")

        magic, version, reserved, total_records = struct.unpack("<4sB3sQ", header)
        
        if magic != b"KMER":
            raise ValueError("无效的文件格式")
        if version != 1:
            raise ValueError(f"不支持的版本号: {version}")

        for _ in range(total_records):
            id_len = struct.unpack("<H", f.read(2))[0]
            id_bytes = f.read(id_len)
            try:
                id_str = id_bytes.decode("utf-8")
            except UnicodeDecodeError:
                id_str = "".join(chr(b) if b < 128 else "_" for b in id_bytes)

            index_count = struct.unpack("<I", f.read(4))[0]
            data = f.read(8 * index_count)
            
            indices = struct.unpack(f"<{index_count}Q", data)
            # logger.debug(
            #     f"Parsed record: id={id_str}, indices={indices}"
            # )
            yield id_str, indices, 0
            rev_indices = [
                i + kmer_count if i < kmer_count else i - kmer_count for i in indices
            ]
            yield id_str, rev_indices, 1
                

def get_feature_matrix(
    ks_file: str,
    precompute_matrix: np.ndarray,
    kmer_count: int
):

    # # Unzip
    # if input_path.endswith(".gz"):
    #     input_unzipped_path = join(globals.temp_dir, os.path.basename(input_path[:-3]))
    #     unzip(input_path, input_unzipped_path)
    # else:
    #     input_unzipped_path = input_path

    # # Convert FASTQ to FASTA
    # if input_unzipped_path.endswith(".fastq") or input_unzipped_path.endswith(".fq"):
    #     input_fasta_path = join(globals.temp_dir, "input.fasta")
    #     convert_fastq_to_fasta(input_unzipped_path, input_fasta_path)
    # elif input_unzipped_path.endswith(".fasta") or input_unzipped_path.endswith(".fa"):
    #     input_fasta_path = input_unzipped_path
    # else:
    #     raise ValueError(
    #         "Unsupported file format. Please provide a FASTA or FASTQ file."
    #     )

    # Get features
    read_names = []
    strands = []
    features = []

    for i, (name, indices, strand) in enumerate(
        parse_kmer_searcher_output(ks_file, kmer_count/2)
    ):
        data = np.ones(len(indices), dtype=np.int8)
        cols = np.array(indices)
        rows = np.zeros(len(indices), dtype=np.int32)
        one_read_feature = csr_matrix((data, (rows, cols)), shape=(1, kmer_count))
        # logger.debug(f"{one_read_feature.shape=}")
        
           
        dr_feature = one_read_feature.dot(precompute_matrix)
        
        # logger.debug(f"{dr_feature.shape=}")
        features.append(dr_feature.toarray())
        read_names.append(name)
        strands.append(strand)

        if (i+1) % 100000 == 0:
            logger.debug(f"Processed {i+1} records.")

    feature_matrix = np.vstack(features)

    logger.debug(
        f"{feature_matrix.shape=}"
    )

    return feature_matrix, read_names, strands


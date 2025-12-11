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
from . import global_variables
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
                

# def get_feature_matrix(
#     ks_file: str,
#     precompute_matrix: np.ndarray,
#     kmer_count: int,
#     read_count: int,
#     embedding_dimensions: int
# ):
#     # Get features
#     read_names = []
#     strands = []
#     feature_matrix = np.empty((read_count*2, embedding_dimensions), dtype=np.float32)
#     for i, (name, indices, strand) in enumerate(
#         parse_kmer_searcher_output(ks_file, kmer_count/2)
#     ):
#         data = np.ones(len(indices), dtype=np.int8)
#         cols = np.array(indices)
#         rows = np.zeros(len(indices), dtype=np.int32)
#         one_read_feature = csr_matrix((data, (rows, cols)), shape=(1, kmer_count))        
           
#         dr_feature = one_read_feature.dot(precompute_matrix)
        
#         # logger.debug(f"{dr_feature.shape=}")
#         feature_matrix[i,:] = dr_feature.toarray()
#         read_names.append(name)
#         strands.append(strand)

#         if (i+1) % 100000 == 0:
#             logger.debug(f"Processed {i+1} records.")

#     return feature_matrix, read_names, strands


def process_read_chunk_simple(task):
    """
    处理一个数据块（多条 reads），返回计算结果和读长信息。
    
    Args:
        task (tuple): (数据块, precompute_matrix, kmer_count)
    """
    chunk_data, precompute_matrix, kmer_count, start_index = task
    
    # 1. 收集数据，准备批量构建稀疏矩阵
    all_data = []
    all_cols = []
    all_rows = [] 
    current_row_in_chunk = 0
    
    # chunk_data 是 (name, indices, strand) 的列表
    for indices in chunk_data:
        # A. 收集数据
        if len(indices) == 0:
            continue
            
        all_data.extend(np.ones(len(indices), dtype=np.int8))
        all_cols.extend(indices)
        all_rows.extend([current_row_in_chunk] * len(indices))
        
        current_row_in_chunk += 1

    if not all_data:
        return np.empty((0, precompute_matrix.shape[1]), dtype=precompute_matrix.dtype), []

    # 2. 一次性构建稀疏矩阵 A_chunk
    N_chunk = current_row_in_chunk
    A_chunk = csr_matrix((all_data, (all_rows, all_cols)), 
                         shape=(N_chunk, kmer_count))
    
    # 3. 一次性进行点积运算
    R_chunk = A_chunk.dot(precompute_matrix) 
    logger.debug(f"processed one batch: {start_index=}")

    # 4. 返回密集数组和读长信息
    return R_chunk.toarray(), start_index

# --- 主程序并行处理 ---
def get_feature_matrix(
    ks_file: str,
    precompute_matrix: np.ndarray,
    kmer_count: int,
    read_count: int,
    chunck_size: int
):
    M = precompute_matrix.shape[1]
    feature_matrix = np.empty((read_count*2, M), dtype=np.float32)
    
    
    # 1. 准备任务块
    all_reads_iterator = parse_kmer_searcher_output(ks_file, kmer_count / 2)
    
    tasks = []
    chunk_data = []
        
    # 划分任务并准备参数元组
    for i, item in enumerate(all_reads_iterator):
        
        chunk_data.append(item[1])
        
        if len(chunk_data) >= chunck_size:
            # 任务元组: (数据块, precompute_matrix, kmer_count)
            task = (chunk_data, precompute_matrix, kmer_count, i + 1 - len(chunk_data))
            tasks.append(task)
            chunk_data = [] # 重置数据块

    # 处理最后一个不完整的块
    if chunk_data:
        task = (chunk_data, precompute_matrix, kmer_count, i + 1 - len(chunk_data))
        tasks.append(task)

    # 2. 创建进程池并启动并行计算
    num_processes = global_variables.threads
    logger.debug(f"Starting parallel processing with {num_processes} processes and {len(tasks)} tasks.")

    
    with Pool(processes=num_processes) as pool:
        # pool.imap_unordered 启动并行计算，无需保持顺序
        # result 是 (R_chunk.toarray(), names_and_strands)
        results = pool.imap_unordered(process_read_chunk_simple, tasks)
        
        # 3. 收集结果
        for feature_array, start_index in results:
            N_chunk = feature_array.shape[0]
            if N_chunk > 0:
                end_index = start_index + N_chunk
                feature_matrix[start_index:end_index, :] = feature_array
                    
    return feature_matrix

def get_metadata(ks_file: str, kmer_count: int):
    read_names = []
    strands = []
    all_reads_iterator = parse_kmer_searcher_output(ks_file, kmer_count / 2)
    for item in all_reads_iterator:
        read_names.append(item[0])
        strands.append(item[2])
    return read_names,strands
import gzip, json, collections, gc, os
from typing import Sequence, Mapping, Collection, Optional, Iterable
from os.path import join
from multiprocessing import Pool, Manager, Array
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
import ctypes

from .fastx_io import (
    FastqLoader,
    FastaLoader,
    get_reverse_complement_record,
    FastxRecord
)
from . import global_variables
from .custom_logging import logger
from .align import Seq

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
                


# --- 1. 定义全局共享变量和初始化函数 ---

# 全局变量，用于在每个子进程中存储 precompute_matrix 的视图
global_precompute_matrix = None
global_kmer_count = None


def init_worker(shared_data_base, shared_indices_base, shared_indptr_base, matrix_shape, kmer_count):
    global global_precompute_matrix
    global global_kmer_count
    
    data_view = np.frombuffer(shared_data_base, dtype=np.float32)
    
    indices_view = np.frombuffer(shared_indices_base, dtype=np.int64)
    indptr_view = np.frombuffer(shared_indptr_base, dtype=np.int64)
    
    global_precompute_matrix = csr_matrix(
        (data_view, indices_view, indptr_view), 
        shape=matrix_shape
    )
    
    global_kmer_count = kmer_count
    
def process_read_chunk_optimized(task):
    chunk_data, start_index = task
    
    global global_precompute_matrix 
    global global_kmer_count
    
    precompute_matrix = global_precompute_matrix
    kmer_count = global_kmer_count
    num_reads_in_chunk = len(chunk_data)
    M = precompute_matrix.shape[1]

    # 1. 收集数据
    all_data = []
    all_cols = []
    all_rows = [] 
    
    for row_idx, indices in enumerate(chunk_data):
        if len(indices) > 0:
            all_data.extend(np.ones(len(indices), dtype=np.int8))
            all_cols.extend(indices)
            all_rows.extend([row_idx] * len(indices))
    
    # 2. 如果整个块都是空的，返回全零矩阵以保持索引对齐
    if not all_data:
        return np.zeros((num_reads_in_chunk, M), dtype=precompute_matrix.dtype), start_index

    # 3. 构建稀疏矩阵，显式指定 shape 以包含空行
    A_chunk = csr_matrix((all_data, (all_rows, all_cols)), 
                         shape=(num_reads_in_chunk, kmer_count))
    
    # 4. 点积运算
    R_chunk = A_chunk.dot(precompute_matrix) 
    
    if start_index % 100000 == 0 and start_index != 0:
        logger.debug(f"{start_index} sequences have been processed.")

    # 5. 返回稠密矩阵（toarray 后形状为 (num_reads_in_chunk, M)）
    return R_chunk.toarray(), start_index

def get_read_length(fasta_file):
    read_length_dict = {}
    for record in FastaLoader(fasta_file):
        read_length_dict[record.name] = len(record.sequence)
    return read_length_dict

def get_feature_matrix(
    ks_file: str,
    fasta_file: str,
    precompute_matrix,
    kmer_count: int,
    read_count: int,
    chunk_size: int
):
    M = precompute_matrix.shape[1]
    feature_matrix = np.empty((read_count*2, M), dtype=np.float32) # read_count 应该是任务数/读数

# 1. 提取 CSR 矩阵的底层数组
    precompute_matrix_csr = precompute_matrix.tocsr() 
    data = precompute_matrix_csr.data
    indices = precompute_matrix_csr.indices
    indptr = precompute_matrix_csr.indptr
    shape = precompute_matrix_csr.shape

    # data (np.float32 -> ctypes.c_float)
    # 从 NumPy 数组直接创建共享数组，避免手动写入
    shared_data = Array(ctypes.c_float, data, lock=False) 

    # indices (np.int64 -> ctypes.c_longlong)
    shared_indices = Array(ctypes.c_longlong, indices, lock=False) 

    # indptr (np.int64 -> ctypes.c_longlong)
    shared_indptr = Array(ctypes.c_longlong, indptr, lock=False)

    # B. 创建任务生成器 (解决任务列表一次性加载)
    
    # 将任务块的生成改为一个生成器函数
    def task_generator(ks_file, kmer_count, chunk_size):
        all_reads_iterator = parse_kmer_searcher_output(ks_file, kmer_count / 2)
        chunk_data = []
        # 从 0 开始计数，因为它是 feature_matrix 的行索引
        start_index = 0 
        
        for i, item in enumerate(all_reads_iterator):
            # item[1] 是 indices
            chunk_data.append(item[1]) 
            
            if len(chunk_data) >= chunk_size:
                # 任务元组只包含数据块和起始索引
                yield (chunk_data, start_index)
                start_index += len(chunk_data) # 更新起始索引
                chunk_data = [] # 重置数据块

        # 处理最后一个不完整的块
        if chunk_data:
            yield (chunk_data, start_index)
            
    tasks = task_generator(ks_file, kmer_count, chunk_size)

    # C. 启动进程池
    num_processes = global_variables.threads 
    logger.debug(f"Starting parallel processing with {num_processes} processes.")

    # 进程池初始化参数：共享内存对象, 数组形状, kmer_count
    init_args = (
        shared_data, 
        shared_indices, 
        shared_indptr, 
        shape, # Python tuple, 自动序列化
        kmer_count
    )
    with Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
        # ⚠️ 使用 imap_unordered 接受生成器作为输入
        # 进程会按需从 tasks 生成器中获取任务，不会一次性加载所有任务
        results = pool.imap_unordered(process_read_chunk_optimized, tasks)
        
        # 3. 收集结果
        for feature_array, start_index in results:
            N_chunk = feature_array.shape[0]
            if N_chunk > 0:
                end_index = start_index + N_chunk
                feature_matrix[start_index:end_index, :] = feature_array
                        
    return feature_matrix


def get_metadata(ks_file: str, fasta_file: str, kmer_count: int):
    all_reads_iterator = parse_kmer_searcher_output(ks_file, kmer_count / 2)
    read_length = {}
    encoded_reads = {}
    for record in FastaLoader(fasta_file):
        read_length[record.name] = len(record.sequence)
    for i, item in enumerate(all_reads_iterator):
        encoded_reads[i] = Seq(
            elements=item[1],
            length=read_length[item[0]],
            read_id=i,
            read_name=item[0],
            strand=item[2]
        )
    return encoded_reads
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
from math import ceil
import math
import time
from dataclasses import dataclass, field
from typing import Sequence, Collection, Mapping, Any, Type, MutableMapping
import psutil
import numpy as np
import sharedmem
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse._csr import csr_matrix

from .custom_logging import logger

import Aligner as cAligner  # type: ignore

def get_new_element():
    return 0


@dataclass(init=False)
class Seq:
    elements: Sequence[int] # kmer indice
    length: int # 使用真实length
    weights: Sequence[float] | None = None
    read_id: int | None = None
    read_name: str
    scale: float
    strand: int = 0

    def __init__(self, elements, length, read_id, read_name, strand, weights=None):
        if len(elements) == 0:
            raise ValueError()
        
        self.elements = elements
        self.length = length
        self.read_id = read_id
        self.read_name = read_name
        self.strand = strand
        self.scale = self.length/len(self.elements)
        
        if weights is None:
            # 自动计算 weights ； 仅供测试使用
            self.weights = np.ones(len(self.elements)).tolist()
        else:
            self.weights = weights


    def __getitem__(self, indices: Sequence[int]) -> "Seq":
        sample_elements = [self.elements[i] for i in indices]
        sample_weights = [self.weights[i] for i in indices]
        return type(self)(
            length=self.length,
            elements=sample_elements,
            weights=sample_weights,
        )

    def sample(self, elements: Collection[int]) -> "Seq":
        indices = [i for i, x in enumerate(self.elements) if x in elements]
        return self[indices]

    def __len__(self):
        return len(self.elements)

@dataclass
class AlignmentResult:
    start_1: int
    end_1: int
    start_2: int
    end_2: int
    identity: float
    match_base_num: int
    match_length: int
    mapq: int
    
    @classmethod
    def from_empty(cls):
        return cls(0, [], [], None, None)

    def __iter__(self):
        yield self.start_1
        yield self.end_1
        yield self.start_2
        yield self.end_1
        yield self.identity
        yield self.match_base_num
        yield self.match_length
        yield self.mapq
        
    # @property
    # def has_traceback(self) -> bool:
    #     if self.score <= 0:  # No traceback needed
    #         return True
    #     if len(self.indices_1) > 0 and len(self.indices_2) > 0:
    #         return True
    #     return False


@dataclass
class _PairwiseAligner:
    
    seq1: Seq
    seq2: Seq
    kmer_size: int

    @staticmethod
    def _get_cell_score(dp_matrix, i, j, x1, x2, w1, w2):
        # match
        if x1 == x2:
            match_score = dp_matrix[i - 1, j - 1] + max(w1, w2)
        else:
            match_score = dp_matrix[i - 1, j - 1] - max(w1, w2)

        # gaps
        gap1_score = dp_matrix[i - 1, j] - w1
        gap2_score = dp_matrix[i, j - 1] - w2

        max_score = max(match_score, gap1_score, gap2_score)

        if match_score == max_score:
            arrow = 1
        elif gap1_score == max_score:
            arrow = 2
        else:
            arrow = 3

        return max_score, arrow

    @staticmethod
    def _dp_traceback(arrow_matrix, i, j):
        aln1, aln2 = [], []
        while i > 0 and j > 0:
            arrow = arrow_matrix[i, j]
            if arrow == 1:
                aln1.append(i - 1)
                aln2.append(j - 1)
                i -= 1
                j -= 1
            elif arrow == 2:
                aln1.append(i - 1)
                aln2.append(-1)
                i -= 1
            elif arrow == 3:
                aln1.append(-1)
                aln2.append(j - 1)
                j -= 1
            else:
                raise ValueError()

        aln1.reverse()
        aln2.reverse()
        return aln1, aln2

    def align(self):
        raise NotImplementedError()


class cWeightedSemiglobalAligner(_PairwiseAligner):
    """
    C++ implementation of `WeightedSemiglobalAligner`
    """

    def align(self, traceback=True,  max_cells=int(1e9)) -> AlignmentResult:
        c_aligner = cAligner.AlignerWrapper()
        seq1, seq2 = self.seq1, self.seq2
        if len(seq1) * len(seq2) > max_cells:
            return AlignmentResult.from_empty()
        weights_1 = seq1.weights
        weights_2 = seq2.weights
        s1, s2 = seq1.elements, seq2.elements
        try:
            score, offset, aln1, aln2 = c_aligner.align_I(
                s1, s2, weights_1, weights_2, traceback
            )
            align_result = get_rough_stats_by_position(
                seq1, seq2, aln1, aln2, self.kmer_size
                )
        
        except Exception:
            raise RuntimeError(f"Alignment failed. {seq1.read_id=} {seq2.read_id=}")
        return align_result
    
    
def wait_for_memory(
    min_free_memory_gb, *, mean_step_wait_seconds=10, max_total_wait_seconds=30
):
    """
    Block until at least `min_free_memory_gb` GB of memory is free or `max_wait_time` seconds have passed.

    Parameters:
    - min_free_memory_gb (float): The minimum amount of free memory in GB that we are waiting for.
    - max_wait_time (float): The maximum amount of time in seconds to wait.

    Returns:
    - bool: True if the required memory is available, False if the max wait time elapsed.
    """
    start_time = time.time()
    min_free_memory_bytes = min_free_memory_gb * (1024**3)
    total_memory = psutil.virtual_memory().total
    if total_memory < min_free_memory_bytes:
        return False

    while True:
        # Check the available memory
        available_memory = psutil.virtual_memory().available

        if available_memory >= min_free_memory_bytes:
            # print(f"Enough memory available: {available_memory / (1024 ** 3):.2f} GB")
            return True

        # Check if the maximum wait time has passed
        elapsed_time = time.time() - start_time
        if elapsed_time > max_total_wait_seconds:
            # print("Maximum wait time elapsed.")
            return False

        # Wait a bit before checking again
        time.sleep(np.random.random() * 2 * mean_step_wait_seconds)


def run_multiprocess_alignment(
    candidates: Sequence[tuple[int, int]],
    encoded_reads: Mapping[int, Seq],
    marker_weights: Mapping[int, int]| None = None,
    *,
    kmer_size: int,
    aligner: Type[_PairwiseAligner],
    processes=4,
    batch_size=100,
    max_total_wait_seconds=120,
    mean_step_wait_seconds=None,
    shuffle=True,
    seed=1,
) -> MutableMapping[tuple[int, int], AlignmentResult]:
    if mean_step_wait_seconds is None:
        mean_step_wait_seconds = processes

    # Remove duplicated candidates
    candidates = list(set(tuple(sorted(x)) for x in candidates))  # type: ignore

    if shuffle:
        candidates = list(candidates).copy()
        np.random.default_rng(seed).shuffle(candidates)

    candidate_count = len(candidates)

    with sharedmem.MapReduce(np=processes) as pool:

        def align(i):

            k1, k2 = candidates[i]
            seq1 = encoded_reads[k1]
            seq2 = encoded_reads[k2]
            if seq1.read_id != k1 or seq2.read_id != k2:
                raise ValueError("Read index doesn't match!")

            min_free_memory_gb = 0.5 * len(seq1) * len(seq2) / 1e9 + 1
            wait_for_memory(
                min_free_memory_gb=min_free_memory_gb,
                max_total_wait_seconds=max_total_wait_seconds,
                mean_step_wait_seconds=mean_step_wait_seconds,
            )
            result = aligner(seq1, seq2, kmer_size=kmer_size).align()  # type: ignore
            return result

        def work(i0):
            output_size = candidate_count
            results = []
            for i in range(i0, i0 + batch_size):
                if i >= output_size:
                    break
                result = align(i)
                results.append(result)
            return i0, results

        finished = 0
        alignment_dict = {}
        previous_progress = 0
        start_time = time.time()

        def reduce(i0, results):
            nonlocal alignment_dict
            
            for idx, i in enumerate(range(i0, i0 + batch_size)):
                if i >= candidate_count:
                    break
                k1, k2 = candidates[i]
                result = results[idx] # 从列表中取出对应位置的结果
                if result != -1:
                    alignment_dict[(k1, k2)] = result[0]
                    alignment_dict[(k2, k1)] = result[1]


            nonlocal finished
            nonlocal previous_progress

            finished = min(finished + batch_size, candidate_count)

            progress = finished / candidate_count
            if progress - previous_progress >= 0.01:
                delta_time = time.time() - start_time
                speed = finished / delta_time
                logger.debug(f"{progress:.1%}\t{speed:.1f} alignments per second")
                previous_progress = progress

        pool.map(work, range(0, candidate_count, batch_size), reduce=reduce)
        
    identity_list = [x.identity for x in alignment_dict.values()]
    logger.debug("Total alignment score: {}".format(np.sum(identity_list)))
    logger.debug("Mean alignment score: {}".format(np.mean(identity_list)))


    return alignment_dict


def get_overlap_candidates(
    neighbor_indices: NDArray,
    n_neighbors: int,
) -> Sequence[tuple[int, int]]:
    if neighbor_indices.shape[1] < n_neighbors:
        raise ValueError("Not enough neighbors in `neighbor_indices`.")
    overlap_candidates = []

    for i1, row in enumerate(neighbor_indices):
        row = row[(row >= 0) & (row != i1)]
        overlap_candidates += [(i1, i2) for i2 in row[:n_neighbors]]

    return overlap_candidates

import math
import numpy as np

def calculate_identity(jaccard, k):
    if jaccard <= 0:
        return 0.0
    # 防止 log 溢出，并处理三代 read 的 Jaccard 修正
    # 公式: I = (2j / (1+j))^(1/k)
    try:
        val = (2 * jaccard) / (jaccard + 1)
        # 对数域计算更稳定
        log_identity = (1 / k) * math.log(val)
        return math.exp(log_identity)
    except (ValueError, ZeroDivisionError):
        return 0.0

def get_rough_stats_by_position(seq1, seq2, aln1, aln2, k):
    # 1. 提取有效比对对
    filtered_pairs = [(i, j) for i, j in zip(aln1, aln2) if i != -1 and j != -1]

    if not filtered_pairs:
        return -1 

    # 提取坐标索引
    # 注意：这里假设 index * scale 是碱基起始位置
    idx1, idx2 = zip(*filtered_pairs)
    
    # 2. 映射原始坐标 (碱基尺度)
    r1_start = int(idx1[0] * seq1.scale)
    r1_end   = min(seq1.length, int(idx1[-1] * seq1.scale) + k)
    
    r2_start = int(idx2[0] * seq2.scale)
    r2_end   = min(seq2.length, int(idx2[-1] * seq2.scale) + k)
    
    # 3. 计算比对跨度 (Alignment Span / Length)
    # 取 query 侧跨度作为基准长度
    match_length = r1_end - r1_start
    if match_length <= 0: return None

    # 4. 计算 Jaccard 和 Identity
    # 注意：match_marker 应该是特征向量（如 count vector）
    # 这里假设 match_marker1/2 是通过采样得到的 k-mer 集合的特征表示
    match_marker1 = [seq1.elements[i] for i in idx1]
    match_marker2 = [seq2.elements[j] for j in idx2]
    
    j = len(set(match_marker1)&set(match_marker2))/len(set(match_marker1)|set(match_marker2))
    
    base_identity = calculate_identity(j, k)

    # 5. 计算修正后的匹配碱基数
    # 核心逻辑：匹配碱基数 = 总长度 * 碱基一致性
    match_base_num = int(match_length * base_identity)

    # 6. 估算 MAPQ
    # 这是一个经验公式，模拟 Minimap2 的逻辑：
    # MAPQ 受 match 数量和 Identity 共同影响。
    # 这里简单用 identity 和匹配的采样点比例来估算
    if base_identity > 0:
        # 匹配点越多，Identity 越高，得分越高。最高 60。
        raw_mapq = -10 * math.log10(1 - base_identity + 1e-6) * (len(filtered_pairs) / 10)
        mapq = min(60, max(0, int(raw_mapq)))
    else:
        mapq = 0

    align_result = AlignmentResult(
        start_1=r1_start, 
        end_1=r1_end, 
        start_2=r2_start, 
        end_2=r2_end, 
        identity=base_identity, 
        match_base_num=match_base_num, 
        match_length=match_length, 
        mapq=mapq
    )

    flip_align_result = AlignmentResult(
        start_1=r2_start, 
        end_1=r2_end, 
        start_2=r1_start, 
        end_2=r1_end, 
        identity=base_identity, 
        match_base_num=int((r2_end-r2_start) * base_identity), 
        match_length=r2_end-r2_start, 
        mapq=mapq
    )
    return align_result, flip_align_result


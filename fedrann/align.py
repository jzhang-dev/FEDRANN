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

    def flip(self) -> "AlignmentResult":
        flipped_result = AlignmentResult(
            start_1=self.start_2,
            end_1=self.end_2,
            start_2=self.start_1,
            end_2=self.end_1,
            identity=self.identity,
            match_base_num=self.match_base_num,
            match_length=self.match_length,
            mapq=self.mapq
        )
        return flipped_result

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
            for i in range(i0, i0 + batch_size):
                if i >= output_size:
                    break
                result = align(i)
            return i0, result

        finished = 0
        alignment_dict = {}
        previous_progress = 0
        start_time = time.time()

        def reduce(i0, result):
            nonlocal alignment_dict
            
            for i in range(i0, i0 + batch_size):
                if i >= candidate_count:
                    break
                k1, k2 = candidates[i]
                if result != -1:
                    alignment_dict[(k1, k2)] = result


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

    # Add flipped results
    flipped_results = {}
    for (i1, i2), result in alignment_dict.items():
        flipped_results[(i2, i1)] = result.flip()
    alignment_dict |= flipped_results

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

def get_rough_stats_by_position(seq1: Seq, seq2: Seq, aln1: list[int], aln2: list[int], k: int):
    """
    针对只包含 K-mer 索引且按顺序排列的对齐结果
    aln1, aln2: 软件返回的列表，数字是 K-mer ID，-1 是空位
    """
    filtered_pairs = [(i, j) for i, j in zip(aln1, aln2) if i != -1 and j != -1]

    # 2. 判断是否为空
    if filtered_pairs:
        path1_pos, path2_pos = map(list, zip(*filtered_pairs))
    else:
        return -1
    

    # 3. 映射原始坐标
    # 使用下标 path1_pos[0] 而不是 aln1[path1_pos[0]]
    r1_start = int(path1_pos[0] * seq1.scale)
    r1_end   = min(seq1.length, int(path1_pos[-1] * seq1.scale) + k)
    
    r2_start = int(path2_pos[0] * seq2.scale)
    r2_end   = min(seq2.length, int(path2_pos[-1] * seq2.scale) + k)
    
    assert r1_end <= seq1.length
    assert r2_end <= seq2.length
    # 4. 计算 Identity
    # path1_pos 存储索引，而不是kmer index
    match_marker1 = [val for i, val in enumerate(seq1.elements) if i in path1_pos]
    match_marker2 = [val for i, val in enumerate(seq2.elements) if i in path2_pos]
    
    matches = sum(1 for i in range(len(match_marker1)) if match_marker1[i] == match_marker2[i])
    match_base_num = int(matches * seq1.scale)
    
    aligned_segments = len(path1_pos)
    match_length = int(aligned_segments * seq1.scale)
      
    kmer_identity = (matches / aligned_segments * 100) if aligned_segments > 0 else 0
    base_identity = math.pow(kmer_identity, 1/k)
    
    kmer_id_freq = matches / aligned_segments
    mapq = int(min(60, 60 * kmer_id_freq * (1 - 1/math.exp(matches/5))))
    
    align_result = AlignmentResult(r1_start, r1_end, r2_start, r2_end, base_identity, match_base_num, match_length, mapq)
    
    return align_result
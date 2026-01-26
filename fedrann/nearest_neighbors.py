from dataclasses import dataclass, field
from typing import Sequence, Type, Mapping, Iterable, Literal
from scipy import sparse
from scipy.sparse._csr import csr_matrix
import numpy as np
from numpy.typing import NDArray
import sklearn.neighbors
import pynndescent
import hnswlib
from math import ceil
import time
from scipy.sparse._csr import csr_matrix

@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, ref: np.ndarray, que: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        raise NotImplementedError()


class NNDescent_ava(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        metric="cosine",
        *,
        index_n_neighbors:int=50,
        n_trees: int| None = 300,
        leaf_size: int| None = 200,
        n_iters: int |None = None,
        diversify_prob: float =1,
        pruning_degree_multiplier:float =1.5,
        low_memory: bool = True,
        n_jobs: int | None = 64,
        seed: int | None = 683985,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            data,
            metric=metric,
            n_neighbors=index_n_neighbors,
            n_trees=n_trees,
            leaf_size=leaf_size,
            n_iters=n_iters,
            diversify_prob=diversify_prob,
            pruning_degree_multiplier=pruning_degree_multiplier,
            low_memory=low_memory,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
        assert index.neighbor_graph is not None
        nbr_indices, distances = index.neighbor_graph
        return nbr_indices, distances



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



def hamming_distance(x, y):
    return np.count_nonzero(x != y)


@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, ref: np.ndarray, que: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        raise NotImplementedError()


def generalized_jaccard_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != 1 or y.shape[0] != 1:
        raise ValueError()
    if x.shape[1] != y.shape[1]:
        raise ValueError()

    s = sparse.vstack([x, y])  # TODO: dense
    jaccard_similarity = s.min(axis=0).sum() / s.max(axis=0).sum()
    return jaccard_similarity


class ExactNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        ref: np.ndarray,
        que: np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
        threads: int | None = 64,
    ):
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric=metric, n_jobs=threads
        )
        nbrs.fit(ref)
        nbr_distance, nbr_indices = nbrs.kneighbors(que)
        return nbr_indices, nbr_distance


class NNDescent_qvt(_NearestNeighbors):
    def get_neighbors(
        self,
        target: np.ndarray,
        query: np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
        *,
        n_trees: int = 100,
        low_memory: bool = True,
        threads: int | None = None,
        seed: int | None = 683985,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            target,
            metric=metric,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
            low_memory=low_memory,
            n_jobs=threads,
            random_state=seed,
            verbose=verbose,
        )

        nbr_indices, nbr_distance = index.query(query, k=n_neighbors)

        return nbr_indices, nbr_distance


class NNDescent_ava(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
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
        if index_n_neighbors < n_neighbors:
            raise ValueError(
                f"index_n_neighbors ({index_n_neighbors}) must be greater than or equal to n_neighbors ({n_neighbors})"
            )
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
        _nbr_indices, _ = index.neighbor_graph
        nbr_indices = _nbr_indices[:,:n_neighbors]
        return nbr_indices


class HNSW(_NearestNeighbors):
    def get_neighbors(
        self,
        ref: np.ndarray,
        que: np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        *,
        threads: int | None = None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):

        if sparse.issparse(ref):
            ref = ref.toarray()  # type: ignore
        if sparse.issparse(que):
            que = que.toarray()  # type: ignore
        if metric == "euclidean":
            space = "l2"
        else:
            space = metric

        # Initialize the HNSW index
        p = hnswlib.Index(space=space, dim=ref.shape[1])
        if threads is not None:
            p.set_num_threads(threads)
        p.init_index(max_elements=ref.shape[0], ef_construction=ef_construction, M=M)
        ids = np.arange(ref.shape[0])
        p.add_items(ref, ids)
        p.set_ef(ef_search)
        nbr_indices, nbr_distance = p.knn_query(que, k=n_neighbors)
        return nbr_indices, nbr_distance




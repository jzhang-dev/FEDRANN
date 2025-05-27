from dataclasses import dataclass, field
import mmh3
from functools import lru_cache
import collections
from typing import Sequence, Type, Mapping, Iterable, Literal
from scipy import sparse
from scipy.sparse._csr import csr_matrix
import numpy as np
from numpy import matlib, ndarray
from numpy.typing import NDArray
import sklearn.neighbors
import pynndescent
import hnswlib
import faiss
from math import ceil
import pynear
import time
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping, Literal
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


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


class NNDescent(_NearestNeighbors):
    def get_neighbors(
        self,
        ref: np.ndarray,
        que: np.ndarray,
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
            ref,
            metric=metric,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
            low_memory=low_memory,
            n_jobs=threads,
            random_state=seed,
            verbose=verbose,
        )

        nbr_indices, nbr_distance = index.query(que, k=n_neighbors)

        return nbr_indices, nbr_distance


class ProductQuantization(_NearestNeighbors):
    def get_neighbors(
        self,
        ref: np.ndarray,
        que: np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        *,
        threads: int = 64,
        m=8,
        nbits=8,
        seed=455390,
    ):

        if sparse.issparse(ref) or sparse.issparse(que):
            raise TypeError("ProductQuantization does not support sparse arrays.")
        assert (
            ref.shape[1] == que.shape[1]
        ), "Reference and query data must have the same feature dimensions."

        feature_count = ref.shape[1]
        if feature_count % m != 0:
            new_feature_count = feature_count // m * m
            rng = np.random.default_rng(seed)
            feature_indices = rng.choice(
                feature_count, new_feature_count, replace=False, shuffle=False
            )
            ref = ref[:, feature_indices]
            que = que[:, feature_indices]
        else:
            new_feature_count = feature_count
        faiss.omp_set_num_threads(threads)
        if metric == "euclidean":
            measure = faiss.METRIC_L2
        else:
            measure = faiss.METRIC_INNER_PRODUCT
            ref = np.array(ref, order="C").astype("float32")
            que = np.array(que, order="C").astype("float32")
            faiss.normalize_L2(ref)
            faiss.normalize_L2(que)

        param = f"PQ{m}"
        index = faiss.index_factory(new_feature_count, param, measure)

        index.train(ref)
        index.add(ref)

        nbr_distances, nbr_indices = index.search(que, n_neighbors)
        return nbr_indices, nbr_distances


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


class SimHash(_NearestNeighbors):
    @staticmethod
    def _get_hash_table(
        feature_count: int, repeats: int, seed: int
    ) -> NDArray[np.int8]:
        rng = np.random.default_rng(seed)
        hash_table = rng.integers(
            0, 2, size=(feature_count, repeats * 8), dtype=np.int8
        )
        hash_table = hash_table * 2 - 1
        return hash_table

    @staticmethod
    def get_simhash(
        data: NDArray | csr_matrix, hash_table: NDArray
    ) -> NDArray[np.uint8]:
        simhash = data @ hash_table
        binary_simhash = np.where(simhash > 0, 1, 0).astype(np.uint8)
        packed_simhash = np.packbits(binary_simhash, axis=-1)
        return packed_simhash

    def get_neighbors(
        self,
        ref: np.ndarray,
        que: np.ndarray,
        n_neighbors: int,
        threads: int = 10,
        repeats=400,
        seed=20141025,
    ):
        assert ref.shape != () and que.shape != ()
        data = sparse.vstack([ref, que])
        assert data.shape is not None
        kmer_num = data.shape[1]
        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)
        simhash = self.get_simhash(data, hash_table)  # type: ignore
        ref_sim = simhash[: ref.shape[0]]
        que_sim = simhash[ref.shape[0] :]
        vptree = pynear.VPTreeBinaryIndex()
        vptree.set(ref_sim)
        vptree_indices, vptree_distances = vptree.searchKNN(que_sim, n_neighbors + 1)
        nbr_indices = np.array(vptree_indices)[:, :-1][:, ::-1]
        nbr_distance = np.array(vptree_distances)[:, :-1][:, ::-1]
        return nbr_indices, nbr_distance

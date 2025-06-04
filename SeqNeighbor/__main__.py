from typing import (
    Optional,
    MutableMapping,
    Any,
    Mapping,
    Sequence,
    Literal,
    List,
    Tuple,
)
import multiprocessing
import gc
import argparse
from itertools import chain
import os
from os.path import join
from math import floor, ceil
from os.path import abspath, join, splitext
import scipy as sp
import pandas as pd
from scipy.sparse._csr import csr_matrix
import json, pickle
import numpy as np
from numpy.typing import NDArray
import logging

from . import __version__, __description__

from .feature_extraction import (
    get_feature_matrix_1,
)
from .preprocess import tf_transform, idf_transform
from .dim_reduction import (
    SpectralEmbedding,
    PCA,
    GaussianRandomProjection,
    SparseRandomProjection,
    mp_SparseRandomProjection,
    scBiMapEmbedding,
)
from .nearest_neighbors import (
    ExactNearestNeighbors,
    NNDescent_ava,
    HNSW,
)
from . import globals
from .custom_logging import logger


logger.setLevel(logging.DEBUG)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input FASTQ/FASTA file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        type=str,
        required=False,
        default="IDF",
        help="Preprocess method you want to implement to matrix.(TF/IDF/TF-IDF/None/count)",
    )
    parser.add_argument(
        "-k",
        "--kmer-size",
        type=int,
        required=False,
        default=16,
        help="K-mer size for feature extraction.",
    )
    parser.add_argument(
        "--kmer-sample-fraction",
        type=float,
        required=False,
        default=0.005,
        help="Percentage of k-mer used to build feature matrix.",
    )
    parser.add_argument(
        "--kmer-min-multiplicity",
        type=int,
        required=False,
        default=2,
        help="Minimum allowed frequency of a k-mer in all reads.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "-d",
        "--dimension-reduction",
        type=str,
        required=False,
        default="SRP",
        help="Dimension reduction method",
    )
    parser.add_argument(
        "-n",
        "--embedding-dimension",
        type=int,
        required=False,
        default=500,
    )
    parser.add_argument(
        "--knn",
        type=str,
        required=False,
        default="NNDescent",
    )
    parser.add_argument(
        "--nndescent-n-trees",
        type=int,
        default=300,
        help="Number of trees to use in NNDescent.",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        required=False,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=356115,
        help="Random seed for reproducibility.",
    )

    # 解析参数
    args = parser.parse_args()
    return args


def get_feature_weights(feature_matrix: csr_matrix, method: str) -> csr_matrix:
    logger.debug(f"Applying preprocessing method {method!r} to feature matrix.")

    if method == "IDF":
        feature_matrix[feature_matrix > 0] = 1
        feature_matrix, _ = idf_transform(feature_matrix)
    elif method == "TF-IDF":
        feature_matrix = tf_transform(feature_matrix)
        feature_matrix, _ = idf_transform(feature_matrix)
    elif method == "None":
        feature_matrix[feature_matrix > 0] = 1
    elif method == "count":
        pass
    elif method == "TF":
        feature_matrix = tf_transform(feature_matrix)
    else:
        raise ValueError(
            f"Invalid preprocess method: {method}. Expected one of TF/IDF/TF-IDF/None/count."
        )
    return feature_matrix


def compute_dimension_reduction(
    feature_matrix: csr_matrix, embedding_dimension: int, method: str
) -> NDArray:
    if method.lower() == "mpsrp":
        logger.info("Using multiprocess Sparse Random Projection for dimensionality reduction.")
        seed = globals.seed + 5599
        embeddings = mp_SparseRandomProjection().transform(
            data=feature_matrix,
            n_dimensions=embedding_dimension,
            seed=seed,
            threads=globals.threads,
        )
    elif method.lower() == "srp":
        logger.info("Using Sparse Random Projection for dimensionality reduction.")
        seed = globals.seed + 5599
        embeddings = SparseRandomProjection().transform(
            data=feature_matrix,
            n_dimensions=embedding_dimension,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dimension reduction method: {method}.")
    logger.debug(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


def get_neighbors_ava(
    embedding_matrix: NDArray, method: str, neighbor_count: int, nndescent_n_trees: int, leaf_size: int = 200
) -> NDArray:
    if method.lower() == "nndescent":
        logger.info(f"Using NNDescent method to find nearest neighbors (n_trees = {nndescent_n_trees}, left_size = {leaf_size})")
        neighbor_indices = NNDescent_ava().get_neighbors(
            embedding_matrix,
            metric="cosine",
            n_neighbors=neighbor_count + 1,  # +1 to include self in the neighbors
            index_n_neighbors=50,
            n_trees=nndescent_n_trees,
            leaf_size=leaf_size,
            n_iters=None,
            diversify_prob=1.0,
            pruning_degree_multiplier=1.5,
            low_memory=True,
            n_jobs=globals.threads,
            seed=globals.seed,
            verbose=True,
        )
    elif method.lower() == "hnsw":
        logger.info("Using HNSW method to find nearest neighbors.")
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'nndescent' or 'hnsw'.")
    return neighbor_indices


def get_output_dataframe(
    neighbor_matrix: NDArray,
    read_names: List[str],
    strands: list[int],
) -> pd.DataFrame:
    query_names = []
    target_names = []
    ranks = []
    query_orientations = []
    target_orientations = []

    for query_index in range(0, neighbor_matrix.shape[0]):
        query_name = read_names[query_index]
        neighbors = neighbor_matrix[query_index]
        query_orientation = ["+", "-"][strands[query_index]]
        for rank, target_index in enumerate(neighbors):
            if target_index == query_index:
                continue
            target_name = read_names[target_index]
            target_orientation = ["+", "-"][strands[target_index]]
            query_names.append(query_name)
            query_orientations.append(query_orientation)
            target_names.append(target_name)
            target_orientations.append(target_orientation)
            ranks.append(rank)

    columns = {
        "query_name": query_names,
        "query_orientation": query_orientations,
        "target_name": target_names,
        "target_orientation": target_orientations,
        "neighbor_rank": ranks,
    }
    df = pd.DataFrame(columns)
    logger.debug(f"Output DataFrame shape: {df.shape}")
    return df


def main():
    args = parse_command_line_arguments()
    globals.threads = args.threads
    globals.seed = args.seed

    output_dir = abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    globals.output_dir = output_dir

    temp_dir = join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    globals.temp_dir = temp_dir

    logger.info(f"SeqNeighbor version: {__version__}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("Starting feature extraction")
    feature_matrix, read_names, strands = get_feature_matrix_1(
        input_path=args.input,
        k=args.kmer_size,
        sample_fraction=args.kmer_sample_fraction,
        min_multiplicity=args.kmer_min_multiplicity,
    )

    feature_matrix = get_feature_weights(feature_matrix, args.preprocess)

    logger.info("Starting dimensionality reduction")
    embedding_matrix = compute_dimension_reduction(
        feature_matrix,
        embedding_dimension=args.embedding_dimension,
        method=args.dimension_reduction,
    )
    del feature_matrix
    gc.collect()

    logger.info("Computing nearest neighbors")
    neighbor_matrix = get_neighbors_ava(
        embedding_matrix,
        method=args.knn,
        neighbor_count=args.neighbor_count,
        nndescent_n_trees=args.nndescent_n_trees,
    )
    del embedding_matrix
    gc.collect()

    logger.info("Saving output")
    output_file = join(output_dir, "overlaps.tsv")
    df = get_output_dataframe(
        neighbor_matrix=neighbor_matrix, read_names=read_names, strands=strands
    )
    df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Output saved to {output_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

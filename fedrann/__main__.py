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
import subprocess
import functools
import gc
from shutil import rmtree
import argparse
from itertools import chain
import os
from os.path import join
from os.path import abspath, join, splitext
import scipy as sp
import numpy as np
import pandas as pd
from scipy.sparse._csr import csr_matrix
from memory_profiler import memory_usage
from numpy.typing import NDArray
import logging

from . import __version__, __description__

from .feature_extraction import (
    get_feature_matrix,
)
from .count_kmers import run_kmer_searcher
from .precompute import get_precompute_matrix
from .nearest_neighbors import (
    ExactNearestNeighbors,
    NNDescent_ava,
    HNSW,
)
from . import global_variables
from .custom_logging import logger, add_log_file


logger.setLevel(logging.DEBUG)


def which(command):
    """
    Python implementation that calls system 'which' command.

    Args:
        command (str): The command to locate

    Returns:
        str: Full path to the executable if found, None otherwise
    """
    try:
        result = subprocess.run(
            ["which", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


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
    # parser.add_argument(
    #     "-p",
    #     "--preprocess",
    #     type=str,
    #     required=False,
    #     default="IDF",
    #     help="Preprocess method you want to implement to matrix.(TF/IDF/TF-IDF/None/count)",
    # )
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
    # parser.add_argument(
    #     "-d",
    #     "--dimension-reduction",
    #     type=str,
    #     required=False,
    #     default="SRP",
    #     help="Dimension reduction method",
    # )
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
        "--nndescent-n-neighbors",
        type=int,
        default=50,
        help="Number of neighbors to use in building NNDescent index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=356115,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-feature-matrix",
        action="store_true",
        default=False,
        help="Save the feature matrix to a file.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Do not remove intermediate files",
    )
    parser.add_argument(
        "--mprof",
        action="store_true",
        default=False,
        help="Record memory usage.",
    )

    # 解析参数
    args = parser.parse_args()
    return args


def get_neighbors_ava(
    embedding_matrix: NDArray,
    method: str,
    nndescent_n_trees: int,
    nndescent_n_neighbors: int,
    leaf_size: int = 200,
) -> tuple[NDArray, NDArray]:
    if method.lower() == "nndescent":
        logger.info(
            f"Using NNDescent method to find nearest neighbors (n_trees = {nndescent_n_trees}, left_size = {leaf_size})"
        )
        neighbor_indices, distances = NNDescent_ava().get_neighbors(
            embedding_matrix,
            metric="cosine",
            index_n_neighbors=nndescent_n_neighbors,
            n_trees=nndescent_n_trees,
            leaf_size=leaf_size,
            n_iters=None,
            diversify_prob=1.0,
            pruning_degree_multiplier=1.5,
            low_memory=True,
            n_jobs=global_variables.threads,
            seed=global_variables.seed,
            verbose=True,
        )
    elif method.lower() == "hnsw":
        logger.info("Using HNSW method to find nearest neighbors.")
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'nndescent' or 'hnsw'.")
    return neighbor_indices, distances


def get_neighbor_table(
    neighbor_matrix: NDArray,
    neighbor_distances: NDArray,
) -> pd.DataFrame:
    """
    Convert neighbor matrix and distance matrix into a DataFrame of neighbor pairs.

    Args:
        neighbor_matrix: 2D array where each row contains indices of neighbors
        neighbor_distances: 2D array with corresponding distances to neighbors

    Returns:
        DataFrame with columns: query_index, target_index, distance, rank
    """
    # Get shapes and validate inputs
    n_queries, n_neighbors = neighbor_matrix.shape
    if neighbor_distances.shape != (n_queries, n_neighbors):
        raise ValueError(
            "neighbor_matrix and neighbor_distances must have the same shape"
        )

    # Create query indices (repeated for each neighbor)
    query_indices = np.repeat(np.arange(n_queries), n_neighbors)

    # Flatten neighbor matrix and distances
    target_indices = neighbor_matrix.ravel()
    distances = neighbor_distances.ravel()

    # Create ranks (0 to n_neighbors-1 repeated for each query)
    ranks = np.tile(np.arange(n_neighbors), n_queries)

    # Filter out self-matches (where query_index == target_index)
    mask = query_indices != target_indices

    # Create DataFrame from filtered arrays
    nbr_df = pd.DataFrame(
        {
            "query_index": query_indices[mask],
            "target_index": target_indices[mask],
            "distance": distances[mask],
            "rank": ranks[mask],
        }
    )

    return nbr_df


def get_metadata_table(
    read_names: Sequence[str],
    strands: Sequence[int],
) -> pd.DataFrame:
    metadata = {
        "index": np.arange(len(read_names)),
        "read_name": read_names,
        "strand": strands,
    }
    metadata_df = pd.DataFrame(metadata)
    return metadata_df


def run_fedrann_pipeline(
    *,
    input_path: str,
    output_dir: str,
    kmer_size: int,
    kmer_sample_fraction: float,
    kmer_min_multiplicity: int,
    embedding_dimension: int,
    knn: str,
    nndescent_n_trees: int,
    nndescent_n_neighbors: int,
    save_feature_matrix: bool,
    keep_intermediates: bool,
):
    """
    Run the SeqNeighbor pipeline with the specified parameters.
    """
    # Extract features
    logger.info("--- 1. Counter kmers ---")
    kmer_searcher_output_path,kmer_counter_path,n_features = run_kmer_searcher(
        input_path=input_path,
        k=kmer_size,
        sample_fraction=kmer_sample_fraction,
        min_multiplicity=kmer_min_multiplicity
    )
    logger.debug(f"kmer_searcher n_features: {n_features}")
    
    logger.info("--- 2. Generate dimension reduction and IDF matrix ---")
    precompute_mat,n_features = get_precompute_matrix(
        n_components=embedding_dimension,
        counter_file=kmer_counter_path,
        n_features=n_features
    )
    logger.debug(f"get_precompute_matrix n_features: {n_features}")
    
    logger.info("--- 3. Generate feature matrix ---")
    embedding_matrix, read_names, strands = get_feature_matrix(
        ks_file=kmer_searcher_output_path,
        precompute_matrix=precompute_mat,
        kmer_count=n_features
    )

    # Save metadata
    metadata_output_file = join(output_dir, "metadata.tsv")
    logger.info(f"Saved metadata table to {metadata_output_file}")
    metadata_df = get_metadata_table(
        read_names=read_names,
        strands=strands,
    )
    metadata_df.to_csv(metadata_output_file, sep="\t", index=False)
    del read_names, strands
    gc.collect()
    

    # Nearest neighbors search
    logger.info("--- 4. Nearest Neighbors Search ---")
    neighbor_matrix, distances = get_neighbors_ava(
        embedding_matrix,
        method=knn,
        nndescent_n_trees=nndescent_n_trees,
        nndescent_n_neighbors=nndescent_n_neighbors,
    )
    del embedding_matrix
    gc.collect()

    # Save output
    nbr_output_file = join(output_dir, "overlaps.tsv")
    logger.debug("Saving overlap table to %s", nbr_output_file)

    nbr_df = get_neighbor_table(
        neighbor_matrix=neighbor_matrix,
        neighbor_distances=distances,
    )
    del neighbor_matrix, distances
    gc.collect()
    nbr_df.to_csv(nbr_output_file, sep="\t", index=False)

    if not keep_intermediates:
        logger.debug("Removing intermediate files")
        rmtree(global_variables.temp_dir)
        
    logger.info(f"Pipeline completed.")


def main():
    args = parse_command_line_arguments()
    globals.threads = args.threads
    globals.seed = args.seed

    if not which("kmer_searcher"):
        raise RuntimeError("Unable to find 'kmer_searcher' executable.")

    output_dir = abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    global_variables.output_dir = output_dir

    logfile = join(output_dir, "fedrann.log")
    add_log_file(logfile)

    temp_dir = join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    global_variables.temp_dir = temp_dir

    logger.info(f"FEDRANN version: {__version__}")
    logger.debug(f"Input file: {args.input}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Parameters: {args}")

    f: functools.partial[None] = functools.partial(
        run_fedrann_pipeline,
        input_path=args.input,
        output_dir=output_dir,
        kmer_size=args.kmer_size,
        kmer_sample_fraction=args.kmer_sample_fraction,
        kmer_min_multiplicity=args.kmer_min_multiplicity,
        embedding_dimension=args.embedding_dimension,
        knn=args.knn,
        nndescent_n_trees=args.nndescent_n_trees,
        nndescent_n_neighbors=args.nndescent_n_neighbors,
        keep_intermediates=args.keep_intermediates,
        save_feature_matrix=args.save_feature_matrix,
    )
    if args.mprof:
        logger.debug("Attention: Memory profiling enabled. Running with memory profiler.")
        mprof_dir = join(output_dir, "mprof")
        os.makedirs(mprof_dir, exist_ok=True)
        mprof_output_path = join(mprof_dir, "memory_profile.dat")
        
        # 确保函数有足够的执行时间
        @memory_usage(
            backend="psutil",
            interval=1,
            multiprocess=True,
            include_children=True,
            timestamps=True,
            max_usage=False,
            stream=open(mprof_output_path, "wt")  # 直接传入文件流
        )
        def profiled_function():
            return f()
        
        # 执行并确保文件关闭
        try:
            profiled_function()
        finally:
            # 确保文件正确关闭
            if 'profiled_function' in locals():
                # 获取stream并关闭
                pass
    else:
        f()
        


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

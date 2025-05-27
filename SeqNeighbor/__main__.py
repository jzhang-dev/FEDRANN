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
import argparse
from itertools import chain
import os
from os.path import join
from math import floor, ceil
from os.path import abspath, join, splitext
import subprocess
import pkg_resources
import time
import scipy as sp
import json, pickle
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from numpy import matlib, ndarray
from . import __version__, __description__
import logging
import colorlog

from .kmer_encoding import (
    load_reads,
    build_sparse_matrix_multiprocess,
)
from .preprocess import manual_tf, manual_idf
from .dim_reduction import (
    SpectralEmbedding,
    PCA,
    UMAPEmbedding,
    GaussianRandomProjection,
    SparseRandomProjection,
    mp_SparseRandomProjection,
    scBiMapEmbedding,
)
from .nearest_neighbors import (
    ExactNearestNeighbors,
    NNDescent,
    HNSW,
    ProductQuantization,
    SimHash,
)
from .format_output import format_output


log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

logger = logging.getLogger("colorlog")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

console_formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=log_colors_config,
)

console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")

    # 添加 mode1 的子解析器
    mode1_parser = subparsers.add_parser(
        "encode",
        help="Encode the input FASTA file into a numerical matrix using K-mer frequencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode1_parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Path to the query (gzip) FASTA file.",
    )  ##支持gzip/非gzip fasta文件
    mode1_parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        help="Path to the target (gzip) FASTA file.",
    )
    mode1_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        default="./encode_dir",
        help="Directory to save output files.",
    )
    mode1_parser.add_argument(
        "-p",
        "--preprocess",
        type=str,
        required=False,
        default="IDF",
        help="Preprocess method you want to implement to matrix.(TF/IDF/TF-IDF/None/count)",
    )
    mode1_parser.add_argument(
        "-k", "--kmer-size", type=int, required=False, default=16, help="K-mer size."
    )
    mode1_parser.add_argument(
        "--kmer-sample-fraction",
        type=float,
        required=False,
        default=0.005,
        help="Percentage of k-mer used to build feature matrix.",
    )
    mode1_parser.add_argument(
        "--kmer-sample-seed",
        type=int,
        required=False,
        default=356115,
        help="Seed used for randomly sampling k-mers.",
    )
    mode1_parser.add_argument(
        "--kmer-min-multiplicity",
        type=int,
        required=False,
        default=2,
        help="Minimum allowed frequency of a k-mer in all reads.",
    )
    mode1_parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Threads numbers used for reads encoding.",
    )

    mode2_parser = subparsers.add_parser(
        "qvt",
        help="Finding Neighbors of query sequences in target sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode2_parser.add_argument(
        "-i", "--encode-dir", type=str, required=True, help="Path to encode directory."
    )
    mode2_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="neighbor_info.tsv",
        help="Path to neighbor information output.",
    )
    mode2_parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default="1",
        help="Threads numbers used for finding neighbors.",
    )
    mode2_parser.add_argument(
        "-d",
        "--dimension-reduction-method",
        type=str,
        required=False,
        default="SparseRP",
        help="Dimension reduction method you want to implement to matrix.(GaussianRP/SparseRP/scBiMap/Spectural/PCA/UMAP/None)",
    )
    mode2_parser.add_argument(
        "-n",
        "--n-dimensions",
        type=int,
        required=False,
        default=500,
        help="Dimension of feature vector after dimensionality reduction",
    )
    mode2_parser.add_argument(
        "--dm-config",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON configuration file for specifying dimensionality reduction parameters.",
    )

    mode2_parser.add_argument(
        "--knn-method",
        type=str,
        required=False,
        default="NNDescent",
        help="k-NN method you want to find neighbors.(ExactNearestNeighbors/NNDescent/HNSW/PQ/SimHash)",
    )
    mode2_parser.add_argument(
        "--n-neighbors",
        type=int,
        required=False,
        default=20,
        help="Nearest neighbors number.",
    )
    mode2_parser.add_argument(
        "--knn-config",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON configuration file for specifying k-NN parameters.",
    )
    mode2_parser.add_argument(
        "--matrix-type",
        type=str,
        required=False,
        default="sparse",
        help="Matrix type of matrix in your input directory.(sparse/dense)",
    )
    # 解析参数
    args = parser.parse_args()
    return args


dim_str2func = {
    "GaussianRP": GaussianRandomProjection,
    "SparseRP": mp_SparseRandomProjection,
    "scBiMap": scBiMapEmbedding,
    "Spectural": SpectralEmbedding,
    "PCA": PCA,
    "UMAP": UMAPEmbedding,
    "None": None,
}
knn_str2func = {
    "ExactNearestNeighbors": ExactNearestNeighbors,
    "NNDescent": NNDescent,
    "HNSW": HNSW,
    "PQ": ProductQuantization,
    "SimHash": SimHash,
}


def get_kw(args, kw_type: Literal["dimension_reduction", "knn"]) -> Mapping[str, str]:
    """
    check whether kw file exist
    if exist, load the kw file,
    else, kw is an empty dictionary.
    """
    kw = {}
    if kw_type == "dimension_reduction" and args.dm_config != None:
        logger.info(f"Loading dimension redution config file {args.dm_config}.")
        with open(args.dm_config, "r") as f:
            kw = json.load(f)
    elif kw_type == "knn_config" and args.knn_config != None:
        logger.info(f"Loading KNN method config file {args.knn_config}.")
        with open(args.knn_config, "r") as f:
            kw = json.load(f)
    return kw


def check_files_exist(file_list: List[str]) -> Tuple[bool, List[str]]:
    missing_files = [file for file in file_list if not os.path.exists(file)]
    return (not missing_files, missing_files)


def encode_qvt(args) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    logger.info(f"Loading input target FASTA file: {args.target}")
    fread_names, fread_orientations, fread_sequences = load_reads(args.target)

    logger.info(f"Loading input query FASTA file: {args.query}")
    qread_names, qread_orientations, qread_sequences = load_reads(args.query)

    logger.info("Building feature matrix for target sequences and query sequences.")
    tar_feature_matrix, preset_col_ind = build_sparse_matrix_multiprocess(
        read_sequences=fread_sequences,
        k=args.kmer_size,
        seed=args.kmer_sample_seed,
        sample_fraction=args.kmer_sample_fraction,
        min_multiplicity=args.kmer_min_multiplicity,
        n_processes=args.threads,
    )
    que_feature_matrix, _ = build_sparse_matrix_multiprocess(
        read_sequences=qread_sequences,
        k=args.kmer_size,
        seed=args.kmer_sample_seed,
        sample_fraction=args.kmer_sample_fraction,
        min_multiplicity=args.kmer_min_multiplicity,
        n_processes=args.threads,
        preset_col_ind=preset_col_ind,
    )
    logger.info(
        f"Target feature matrix size: {tar_feature_matrix.shape}, query feature matrix size: {que_feature_matrix.shape}."
    )
    return qread_names, fread_names, tar_feature_matrix, que_feature_matrix


def encode_ava(args) -> Tuple[List[str], np.ndarray]:
    logger.info(f"Loading input target FASTA file: {args.target}")
    fread_names, fread_orientations, fread_sequences = load_reads(args.target)

    logger.info("Building feature matrix for sequences.")
    tar_feature_matrix, _ = build_sparse_matrix_multiprocess(
        read_sequences=fread_sequences,
        k=args.kmer_size,
        seed=args.kmer_sample_seed,
        sample_fraction=args.kmer_sample_fraction,
        min_multiplicity=args.kmer_min_multiplicity,
        n_processes=args.threads,
    )
    logger.info(f"Feature matrix size: {tar_feature_matrix.shape}")
    return fread_names, tar_feature_matrix


def tfidf_qvt(
    tar_feature_matrix: np.ndarray, que_feature_matrix: np.ndarray, preprocess_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
    if preprocess_type == "IDF":
        tar_feature_matrix[tar_feature_matrix > 0] = 1
        que_feature_matrix[que_feature_matrix > 0] = 1
        tar_train, idf = manual_idf(tar_feature_matrix)
        que_fit = que_feature_matrix.multiply(idf)
    elif preprocess_type == "TF-IDF":
        tar_tf = manual_tf(tar_feature_matrix)
        que_idf = manual_tf(que_feature_matrix)
        tar_train, idf = manual_idf(tar_tf)
        que_fit = que_idf.multiply(idf)
    elif preprocess_type == "None":
        tar_feature_matrix[tar_feature_matrix > 0] = 1
        que_feature_matrix[que_feature_matrix > 0] = 1
    elif preprocess_type == "count":
        tar_train = tar_feature_matrix
        que_fit = que_feature_matrix
    elif preprocess_type == "TF":
        tar_train = manual_tf(tar_feature_matrix)
        que_fit = manual_tf(que_feature_matrix)
    else:
        logger.error(
            f"Invalid preprocess method: {preprocess_type}. Expected one of TF/IDF/TF-IDF/None/count."
        )
    return tar_train, que_fit


def tfidf_ava(feature_matrix: np.ndarray, preprocess_type: str) -> np.ndarray:
    if preprocess_type == "IDF":
        feature_matrix[feature_matrix > 0] = 1
        pre_feature_matrix, _ = manual_idf(feature_matrix)
    elif preprocess_type == "TF-IDF":
        _feature_matrix = manual_tf(feature_matrix)
        pre_feature_matrix, _ = manual_idf(_feature_matrix)
    elif preprocess_type == "None":
        feature_matrix[feature_matrix > 0] = 1
    elif preprocess_type == "count":
        pre_feature_matrix = feature_matrix
    elif preprocess_type == "TF":
        pre_feature_matrix = manual_tf(feature_matrix)
    else:
        logger.error(
            f"Invalid preprocess method: {preprocess_type}. Expected one of TF/IDF/TF-IDF/None/count."
        )
    return pre_feature_matrix


def dimension_reduction_qvt(args) -> Tuple[np.ndarray, np.ndarray]:
    dim = args.dimension_reduction_method
    target_fm_file = join(args.encode_dir, "target_feature_matrix.npz")
    query_fm_file = join(args.encode_dir, "query_feature_matrix.npz")
    if args.matrix_type == "sparse":
        query_fm = sp.sparse.load_npz(query_fm_file)
        taget_fm = sp.sparse.load_npz(target_fm_file)
        assert isinstance(
            query_fm, sp.sparse.csr_matrix
        ), "Feature matrix must be a sparse matrix."
        assert isinstance(
            taget_fm, sp.sparse.csr_matrix
        ), "Feature matrix must be a sparse matrix."
    elif args.matrix_type == "dense":
        query_fm = np.load(query_fm_file)["arr_0"]
        taget_fm = np.load(target_fm_file)["arr_0"]
    else:
        logger.error(
            f"Invalid matrix type: {args.matrix_type}. Expected one of sparse/dense."
        )

    if (query_fm.shape[1] > 2**32 or taget_fm.shape[1] > 2**32) and dim in [
        "scBiMap",
        "PCA",
        "Spectural",
    ]:
        logger.error(
            f"Errors may occur using {dim} method on this large feature matrix. Consider using one of the following methods: GaussianRP, SparseRP or UMAP."
        )

    dimension_reduction_kw = get_kw(args, "dimension_reduction")
    if dim == "None":
        logger.info("Dimensionality reduction is explicitly disabled.")
        taget_fm_dim = taget_fm
        query_fm_dim = query_fm
    elif dim not in dim_str2func.keys():
        logger.error(
            f"Invalid dimension reduction method: {dim}. Expected one of {list(dim_str2func.keys())}."
        )
    else:
        dimension_reduction_method = dim_str2func[dim]
        logger.info(f"Performing {dim} dimensionality reduction.")
        if dim in ["GaussianRP", "SparseRP"]:
            query_fm_dim = dimension_reduction_method().transform(
                query_fm, args.n_dimensions, **dimension_reduction_kw
            )
            taget_fm_dim = dimension_reduction_method().transform(
                taget_fm, args.n_dimensions, **dimension_reduction_kw
            )
        else:
            data = sp.sparse.vstack([taget_fm, query_fm])
            data_dim = dimension_reduction_method().transform(
                data, args.n_dimensions, **dimension_reduction_kw
            )
            taget_fm_dim = data_dim[taget_fm.shape[0] :, :]
            query_fm_dim = data_dim[: taget_fm.shape[0], :]
    return taget_fm_dim, query_fm_dim


def dimension_reduction_ava(args) -> np.ndarray:
    dim = args.dimension_reduction_method
    target_fm_file = join(args.encode_dir, "feature_matrix.npz")
    taget_fm = sp.sparse.load_npz(target_fm_file)
    assert isinstance(
        taget_fm, sp.sparse.csr_matrix
    ), "Feature matrix must be a sparse matrix."

    if taget_fm.shape[1] > 2**32 and dim in ["scBiMap", "PCA", "Spectural"]:
        logger.warning(
            f"Errors may occur using {dim} method on this large feature matrix. Consider using one of the following methods: GaussianRP, SparseRP, or UMAP."
        )

    dimension_reduction_kw = get_kw(args, "dimension_reduction")
    if dim == "None":
        logger.info("Dimensionality reduction is explicitly disabled.")
        taget_fm_dim = taget_fm
    elif dim not in dim_str2func.keys():
        logger.error(
            f"Invalid dimension reduction method: {dim}. Expected one of {list(dim_str2func.keys())}."
        )
    else:
        dimension_reduction_method = dim_str2func[dim]
        logger.info(f"Performing {dim} dimensionality reduction.")
        taget_fm_dim = dimension_reduction_method().transform(
            taget_fm, args.n_dimensions, **dimension_reduction_kw
        )
    return taget_fm_dim


def get_neighbor(
    taget_fm_dim: np.ndarray, query_fm_dim: np.ndarray, args
) -> Tuple[np.ndarray, np.ndarray]:
    knn_method = args.knn_method
    if knn_method not in knn_str2func.keys():
        raise KeyError(
            f"Invalid k Neareast Neighbor method: {knn_method}. Expected one of {list(knn_str2func.keys())}."
        )

    nearest_neighbors_kw = get_kw(args, "knn")

    logger.info(f"Using {knn_method} method to find nearest neighbors.")
    nearest_neighbors_method = knn_str2func[knn_method]
    neighbor_indices, distance = nearest_neighbors_method().get_neighbors(
        taget_fm_dim,
        query_fm_dim,
        n_neighbors=args.n_neighbors + 1,
        threads=args.threads,
        **nearest_neighbors_kw,
    )
    return neighbor_indices, distance


def main():
    args = parse_command_line_arguments()
    encode_dir = (
        abspath(args.encode_dir) if args.mode == "qvt" else abspath(args.output_dir)
    )
    fm_file = join(encode_dir, "feature_matrix.npz")
    name_file = join(encode_dir, "read_name.pkl")
    target_fm_file = join(encode_dir, "target_feature_matrix.npz")
    query_fm_file = join(encode_dir, "query_feature_matrix.npz")
    qname_file = join(encode_dir, "query_name.pkl")
    fname_file = join(encode_dir, "target_name.pkl")

    if args.mode == "encode":
        output_dir = abspath(args.output_dir)
        fm_file = join(output_dir, "feature_matrix.npz")
        name_file = join(output_dir, "read_name.pkl")
        target_fm_file = join(output_dir, "target_feature_matrix.npz")
        query_fm_file = join(output_dir, "query_feature_matrix.npz")
        qname_file = join(output_dir, "query_name.pkl")
        fname_file = join(output_dir, "target_name.pkl")

        os.makedirs(output_dir, exist_ok=True)
        if args.target == args.query:
            read_names, feature_matrix = encode_ava(args)
            feature_matrix = tfidf_ava(feature_matrix, args.preprocess)
            logger.info("Saving output to files.")
            with open(name_file, "wb") as file1:
                pickle.dump(read_names, file1)
            assert isinstance(feature_matrix, sp.sparse.csr_matrix)
            sp.sparse.save_npz(fm_file, feature_matrix)
        else:
            qread_names, fread_names, tar_feature_matrix, que_feature_matrix = (
                encode_qvt(args)
            )
            tar_train, que_fit = tfidf_qvt(
                tar_feature_matrix, que_feature_matrix, args.preprocess
            )
            logger.info("Saving output to files.")
            with open(qname_file, "wb") as file1:
                pickle.dump(qread_names, file1)
            with open(fname_file, "wb") as file2:
                pickle.dump(fread_names, file2)
            assert isinstance(target_fm_file, sp.sparse.csr_matrix)
            assert isinstance(tar_train, sp.sparse.csr_matrix)
            sp.sparse.save_npz(que_fit, tar_train)
            sp.sparse.save_npz(query_fm_file, que_fit)

        logger.info(f"SeqNeighbor encoding is complete!")

    elif args.mode == "qvt":
        ## remind unsuitable combination of dimensional-reduction method and knn method
        if args.dimension_reduction_method == "None" and args.knn_method in [
            "HNSW",
            "PQ",
            "NNDescent",
        ]:
            logger.warning(
                f"Using {args.knn_method} in high-dimensional sparse matrix may consume significant time and memory."
            )

        ava_files_exist, ava_missing_files = check_files_exist([fm_file, name_file])
        qvt_files_exist, qvt_missing_files = check_files_exist(
            [target_fm_file, query_fm_file, qname_file, fname_file]
        )

        if ava_files_exist:
            logger.info(
                "All required file exist, starting excuting dimensional reduction and knn neighbor finding."
            )
            fm_dim = dimension_reduction_ava(args)
            neighbor_indices, distance = get_neighbor(fm_dim, fm_dim, args)
            with open(name_file, "rb") as file:
                read_name = pickle.load(file)
            df = format_output(neighbor_indices, distance, read_name, read_name)

        elif qvt_files_exist:
            logger.info(
                "All required file exist, starting excuting dimensional reduction and knn neighbor finding."
            )
            target_fm_dim, query_fm_dim = dimension_reduction_qvt(args)
            neighbor_indices, distance = get_neighbor(target_fm_dim, query_fm_dim, args)
            with open(fname_file, "rb") as file:
                target_read_name = pickle.load(file)
            with open(qname_file, "rb") as file:
                query_read_name = pickle.load(file)
            df = format_output(
                neighbor_indices, distance, target_read_name, query_read_name
            )
        else:
            missing_files = ava_missing_files + qvt_missing_files
            logger.error(f"{missing_files} file missing.")
        df.to_csv(args.output_file, sep="\t", index=False)
        logger.info("SeqNeighbor qvt is complete!")
    else:
        print("No mode selected. Use 'encode' or 'qvt'.")
        exit(1)


if __name__ == "__main__":
    main()

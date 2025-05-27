import gzip, json, collections
from typing import Sequence, Mapping, Collection,Optional
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
from .fastx_io import FastqLoader
import time 
import xxhash
from multiprocessing import Pool
from functools import partial
import gc

def init_reverse_complement():
    TRANSLATION_TABLE = str.maketrans("ACTGactg", "TGACtgac")

    def reverse_complement(sequence: str) -> str:
        """
        >>> reverse_complement("AATC")
        'GATT'
        >>> reverse_complement("CCANT")
        'ANTGG'
        """
        sequence = str(sequence)
        return sequence.translate(TRANSLATION_TABLE)[::-1]

    return reverse_complement


reverse_complement = init_reverse_complement()


def load_reads(fasta_path: str):
    read_sequences = []
    read_names = []
    read_orientations = []
    loader = FastqLoader(file_path=fasta_path)
    for record in loader:  # 迭代获取每条序列
        seq = str(record.sequence)
        read_sequences.append(seq)
        read_names.append(record.name)
        read_orientations.append("+")

        # Include reverse complement
        read_sequences.append(reverse_complement(seq))
        read_names.append(record.name)
        read_orientations.append("-")

    return read_names, read_orientations, read_sequences


def process_sequence(args, k, seed, max_hash,preset_col_ind):
    row_idx, seq = args
    seq_counts = collections.defaultdict(int)
    kmers = (seq[p:p+k] for p in range(len(seq) - k + 1))
    # Count kmers in this sequence
    for kmer in kmers:
        hashed = xxhash.xxh3_64(kmer, seed=seed).intdigest()
        if hashed <= max_hash:
            if preset_col_ind == []:
                seq_counts[hashed] += 1
            else:
                if hashed in preset_col_ind:
                    seq_counts[hashed] += 1
    # if row_idx % 200_000 == 0:
    #     print(row_idx)
    ## print process schedule
    count = len(seq_counts)
    if count == 0:
        return np.empty((0, 3), dtype=np.uint64)
    result = np.empty((count, 3), dtype=np.uint64)
    for i, (hashed, cnt) in enumerate(seq_counts.items()):
        result[i] = [row_idx, hashed, cnt]
    return result

def build_sparse_matrix_multiprocess(read_sequences, k, seed, sample_fraction, min_multiplicity, n_processes, preset_col_ind = []):
    all_kmer_number = 2**64
    max_hash = all_kmer_number * sample_fraction
    # Parallel processing with imap
    with Pool(n_processes,maxtasksperchild=100) as pool:
        func = partial(process_sequence, 
                      k=k, seed=seed, 
                      max_hash=max_hash,
                      preset_col_ind=preset_col_ind)
        
        # Process results incrementally
        row_ind, col_ind, data = [], [], []
        for result in pool.imap(func, enumerate(read_sequences), chunksize=1000):
            if result.size > 0:
                row_ind.append(result[:, 0])
                col_ind.append(result[:, 1])
                data.append(result[:, 2])
        pool.close()
        pool.join()
        gc.collect()

        # Concatenate all results
    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    data = np.concatenate(data)

    # Build sparse matrix
    re_col_ind = pd.factorize(col_ind)[0].tolist()
    n_rows = len(read_sequences)
    n_cols = max(re_col_ind) + 1

    _feature_matrix = sp.csr_matrix(
        (data, (row_ind, re_col_ind)),
        shape=(n_rows, n_cols),
        dtype=np.int32
    )

    # Filter by multiplicity
    col_sums = _feature_matrix.sum(axis=0).A1
    mask = col_sums >= min_multiplicity
    feature_matrix = _feature_matrix[:, mask]
    return feature_matrix,set(col_ind)

def encode_reads(
    fasta_path: str,
    info_path: str,
    k,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
    processes: int,
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads info
    info_df = pd.read_table(info_path).set_index("read_name")

    # Load reads
    read_names, read_orientations, read_sequences = load_reads(fasta_path=fasta_path)

    feature_matrix = build_sparse_matrix_multiprocess(
        read_sequences=read_sequences,
        k=k,
        seed=seed,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        n_processes=processes
    )

    # Build metadata
    def flip(strand):
        return {"+": "-", "-": "+"}[strand]

    rows = []
    for i in range(len(read_sequences)):
        read_name = read_names[i]
        read_orientation = read_orientations[i]
        reference_strand = info_df.at[read_name, "reference_strand"]
        if read_orientation == "-":
            reference_strand = flip(reference_strand)
        rows.append(
            dict(
                read_id=i,
                read_name=read_name,
                read_orientation=read_orientation,
                read_length=info_df.at[read_name, "read_length"],
                reference_strand=reference_strand,
                reference_start=info_df.at[read_name, "reference_start"],
                reference_end=info_df.at[read_name, "reference_end"],
            )
        )
    metadata = pd.DataFrame(rows)

    return feature_matrix, metadata

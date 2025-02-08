from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json, collections
from typing import Sequence, Mapping, Collection,Optional
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
from isal import igzip


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

def open_(path, mode="rt", gzipped: Optional[bool] = None, **kw):
    if gzipped is None:
        gzipped = path.endswith(".gz")
    if gzipped:
        open_ = igzip.open
        return open_(path, mode)
    else:
        open_ = open
    return open_(path, mode, **kw)

def load_reads(fasta_path: str):
    read_sequences = []
    read_names = []
    read_orientations = []
    
    with open_(fasta_path,'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            read_sequences.append(seq)
            read_names.append(record.id)
            read_orientations.append("+")

            # Include reverse complement
            read_sequences.append(reverse_complement(seq))
            read_names.append(record.id)
            read_orientations.append("-")

    return read_names, read_orientations, read_sequences


def build_kmer_index(
    read_sequences: Sequence[str],
    k: int,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
) -> Mapping[str, int]:
    kmer_counter = collections.Counter()
    for seq in read_sequences:
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            kmer_counter[kmer] += 1
    rng = np.random.default_rng(seed=seed)
    vocabulary = set(
        x
        for x, count in kmer_counter.items()
        if count >= min_multiplicity and rng.random() <= sample_fraction
    )
    vocabulary |= set(reverse_complement(x) for x in vocabulary)
    kmer_indices = {kmer: i for i, kmer in enumerate(vocabulary)}
    return kmer_indices


def build_feature_matrix(
    read_sequences: Sequence[str],
    kmer_indices: Mapping[str, int],
    k: int,
) -> tuple[sp.csr_matrix, Sequence[Sequence[int]]]:
    row_ind, col_ind, data = [], [], []
    read_features = []
    for i, seq in enumerate(read_sequences):
        features_i = []
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_indices.get(kmer)
            if j is None:
                continue
            features_i.append(j)

        read_features.append(features_i)

        kmer_counts = collections.Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)

    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(kmer_indices))
    )
    return feature_matrix


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

    # Build vocabulary
    kmer_indices = build_kmer_index(
        read_sequences=read_sequences,
        k=k,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed,
        processes=processes,
    )

    # Build matrix
    feature_matrix, read_features = build_feature_matrix(
        read_sequences=read_sequences,
        kmer_indices=kmer_indices,
        k=k,
        processes=processes,
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

    return feature_matrix, read_features, metadata

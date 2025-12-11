import subprocess
import tempfile
import os
from os.path import abspath, isfile, join
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, Optional, Iterable
import struct
from collections import namedtuple
from typing import Iterator

from .custom_logging import logger
from . import globals
from .fastx_io import (
    unzip,
    convert_fastq_to_fasta,
)


def count_lines(filename):
    """使用 wc -l 命令统计行数"""
    result = subprocess.run(
        ["wc", "-l", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wc failed: {result.stderr}")
    return int(result.stdout.split()[0])


def run_kmer_searcher(
    input_path: str, 
    k: int, 
    sample_fraction: float, 
    min_multiplicity: int = 2
):
    
    if input_path.endswith(".gz"):
        input_unzipped_path = join(globals.temp_dir, os.path.basename(input_path[:-3]))
        unzip(input_path, input_unzipped_path)
    else:
        input_unzipped_path = input_path

    # Convert FASTQ to FASTA
    if input_unzipped_path.endswith(".fastq") or input_unzipped_path.endswith(".fq"):
        input_fasta_path = join(globals.temp_dir, "input.fasta")
        convert_fastq_to_fasta(input_unzipped_path, input_fasta_path)
    elif input_unzipped_path.endswith(".fasta") or input_unzipped_path.endswith(".fa"):
        input_fasta_path = input_unzipped_path
    else:
        raise ValueError(
            "Unsupported file format. Please provide a FASTA or FASTQ file."
        )
        
    jf_path = join(globals.temp_dir, "kmer_counts.jf")
    hash_size = "10G"
    threads = globals.threads

    jellyfish_count_command = [
        "jellyfish",
        "count",
        "-m",
        str(k),
        "-s",
        hash_size,
        "-t",
        str(threads),
        "-C",
        input_fasta_path,
        "-o",
        jf_path,
    ]
    logger.debug(
        f"Running Jellyfish count command: {' '.join(jellyfish_count_command)}"
    )
    subprocess.run(jellyfish_count_command, check=True)
    if not isfile(jf_path):
        raise RuntimeError(f"Jellyfish count output file not found: {jf_path}")

    fwd_kmer_library_path = join(globals.temp_dir, "fwd_kmer_library.fasta")
    rev_kmer_library_path = join(globals.temp_dir, "rev_kmer_library.fasta")

    awk_script = r"""
        BEGIN {
            srand(seed);  
            skip_prob = 1 - p;  
        }
        {
            if (NR % 2 == 1) {
                current_pair = $0;  
                next;              
            } else {
                current_pair = current_pair ORS $0;
                if (rand() > skip_prob) {
                    print current_pair;
                }
            }
        }
    """
    command = f"jellyfish dump -L {min_multiplicity} {jf_path} | awk -v p={sample_fraction} -v seed={globals.seed} '{awk_script}' > {fwd_kmer_library_path}"
    logger.debug(f"Running Jellyfish dump command: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug(f"Counting kmers in the library: {fwd_kmer_library_path}")
    kmer_count = (
        count_lines(fwd_kmer_library_path) // 2
    )  # 每个kmer有两行（header和sequence）
    logger.debug(f"Number of kmers in the library: {kmer_count}")

    command = f"seqkit seq -r -p -t DNA -j {globals.threads} {fwd_kmer_library_path} > {rev_kmer_library_path}"
    logger.debug(f"Creating reverse complement for kmer library: {command}")
    subprocess.run(command, shell=True, check=True)

    kmer_searcher_output_dir = join(globals.temp_dir, "kmer_searcher")

    command = f"cat {fwd_kmer_library_path} {rev_kmer_library_path} | grep -v '^>' | kmer_searcher /dev/stdin {input_fasta_path} {kmer_searcher_output_dir} {k} {globals.threads}"
    logger.debug(f"Searching kmers for forward strands: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug("Parsing kmer_searcher output")
    kmer_searcher_output_path = join(kmer_searcher_output_dir, "output.bin")
    kmer_counter_path = join(kmer_searcher_output_dir, "kmer_frequency.bin") 
    if not isfile(kmer_searcher_output_path):
        raise RuntimeError(
            f"kmer_searcher output file not found: {kmer_searcher_output_path}"
        )
        
    if not isfile(kmer_counter_path):
        raise RuntimeError(
            f"kmer_searcher output file not found: {kmer_counter_path}"
        )
        
    return kmer_searcher_output_path,kmer_counter_path,kmer_count*2
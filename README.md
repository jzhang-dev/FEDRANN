# Fedrann

Fedrann is a scalable pipeline for overlap detection based on large-scale sequencing data. It is based on three steps: **f**eature **e**xtraction, **d**imenality **r**eduction, and **a**pproximate *k*-**n**earest **n**eighbor search. 

[![CI](https://github.com/jzhang-dev/FEDRANN/actions/workflows/ci.yml/badge.svg)](https://github.com/jzhang-dev/FEDRANN/actions/workflows/ci.yml)

## Installation

```bash
# Clone the repository
git clone --recursive git@github.com:jzhang-dev/FEDRANN.git

# Set up Conda environment
conda env create -p ./env -f environment.yml

# Build kmer_searcher
cd kmer_searcher && ./build.sh && cd ..

# Install Fedrann
pip install .

# Test installation
test/test.sh
```

## Usage

Run the pipeline from the command line:

```bash
fedrann \
    # Specify the input file (a FASTQ/FASTA file of sequencing reads).
    -i test/data/reads.fasta.gz \
    # Specify the output directory where results will be saved.
    -o test/output/ \
    # Set the k-mer size to 15 for feature extraction.
    -k 15 \
    # Use only 5% of the total k-mers for feature extraction.
    --kmer-sample-fraction 0.05 \
    # Set the random seed for reproducible results.
    --seed 602 \
    # Ignore k-mers that appear less than 2 times to filter out sequencing errors.
    --kmer-min-multiplicity 2 \
    # Use the mpsrp (multiprocessed sparse random projection) algorithm for dimension reduction.
    --dimension-reduction mpsrp \
    # Save the feature matrix file after processing.
    --save-feature-matrix \
    # Use 32 CPU threads to parallelize the process.
    --threads 32
```

See all options with:

```bash
fedrann --help
```

## Output

Fedrann generates the following output files to help you understand your analysis.

- `fedrann.log`: A log file that details the pipeline's progress, from start to finish.

- `overlaps.tsv`: This file contains a list of every sequence analyzed, including its name and orientation.

- `metadata.tsv`: This file lists all identified candidate overlaps and their similarity metrics.

- `feature_matrix.npz`: (Optional) This sparse-format file contains the feature matrix generated during the analysis.

### `overlaps.tsv`
This file serves as a reference for all input sequences. The index column provides a numerical identifier for each sequence. The `read_name` column contains the original name of the sequence, while the `strand` column specifies its orientation. A strand value of `0` denotes the original sequence, and a value of `1` denotes its reverse complement.

Example `overlaps.tsv` file:
```
index   read_name   strand
0   c2924806-d5c6-4564-b31a-c701c0226fbc    0
1   c29246d8-d6c6-4564-b31a-c701c0226fbc    1
2   b5ec3070-2ba3-430d-a55d-1d7b178c8d36    0
3   b5ec3070-2ba3-430d-a55d-1d7b178c8d36    1
4   5b1d405b-08b4-448d-b601-dd922aa9380c    0
5   5b1d405b-08b4-448d-b601-dd922aa9380c    1
6   f3d50991-ad3e-4564-98de-97bf986f992c    0
7   f3d50991-ad3e-4564-98de-97bf986f992c    1
8   316caa78-7d27-4d49-b0f3-684fa17063e4    0
9   316caa78-7d27-4d49-b0f3-684fa17063e4    1
10  7b2ee773-fc72-4b54-b6fe-a412df3f8744    0
```

### `metadata.tsv`
This file details the candidate overlaps identified by the tool. You can use the `query_index` and `target_index` to look up the full sequence names in the `overlaps.tsv` file.

Example `metadata.tsv` file:
```
query_index target_index    distance    rank
0   6541    0.6878114767745782  1
0   828 0.6921228205223058  2
0   10329   0.7078562118327599  3
0   2847    0.7284434279308162  4
0   9642    0.7328719782924367  5
0   7857    0.7369192193171591  6
1   10328   0.661215875322702   1
1   6540    0.6803799447921752  2
1   5053    0.724938780681381   3
1   2846    0.7418480245039423  4
1   829 0.7578067926101968  5
1   9622    0.7603659996459153  6
```
`distance`: Measures the dissimilarity between the embedded vectors of the query and target sequences. A smaller value indicates higher similarity between the sequences.

`rank`: The similarity rank of the `target_index` sequence among all potential matches for the `query_index`. A lower rank (closer to 1) signifies a better match.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Contact

Authors: Jia-Yuan Zhang, Changjiu Miao  
Email: zhangjiayuan@genomics.cn, miaochangjiu@genomics.cn

## Citation

Zhang, J. Y., Miao, C., Qiu, T., Xia, X., He, L., He, J., ... & Dong, Y. (2025). FEDRANN: effective long-read overlap detection based on dimensionality reduction and approximate nearest neighbors. bioRxiv, 2025-05. https://www.biorxiv.org/content/10.1101/2025.05.30.656979v3





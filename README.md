# Fedrann

Fedrann is a scalable pipeline for overlap detection based on large-scale sequencing data. It is based on three core concepts: **f**eature **e**xtraction, **d**imensionality **r**eduction, and **a**pproximate *k*-**n**earest **n**eighbor search. 

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

- `overlaps.tsv`: This file lists all identified candidate overlaps with sequence names, orientations, and their similarity metrics.

- `feature_matrix.npz`: (Optional) This sparse-format file contains the feature matrix generated during the analysis.

### `overlaps.tsv`
This file details the candidate overlaps identified by the tool. Each row represents a potential overlap between two sequences.

Example `overlaps.tsv` file:
```
query_name	query_orientation	target_name	target_orientation	neighbor_rank	distance
c2924806-d5c6-4564-b31a-c701c0226fbc	+	7e3d8f0e-fcf0-4073-ad6b-6d742245f29b	+	1	0.6606640366100363
c2924806-d5c6-4564-b31a-c701c0226fbc	+	dd1acbc0-f219-4701-b1eb-00b9850d3d9e	-	2	0.6690033249609217
c2924806-d5c6-4564-b31a-c701c0226fbc	+	9b58f527-6e4d-4005-92cb-4863b6d42229	+	3	0.7207873470869667
c2924806-d5c6-4564-b31a-c701c0226fbc	+	932c6e73-e5d9-4bbe-b88f-e4babfc043a1	-	4	0.7347354015555427
c2924806-d5c6-4564-b31a-c701c0226fbc	+	416c786a-11f1-487f-99cc-830a40ee1c6c	-	5	0.7421239213106542
c2924806-d5c6-4564-b31a-c701c0226fbc	+	99640d4e-d299-414c-ade3-ebe567b4d1ef	+	6	0.7520030538483087
c2924806-d5c6-4564-b31a-c701c0226fbc	+	ffdda27d-9d97-4d01-8e61-8a6e281e0f69	+	7	0.7670946853284661
c2924806-d5c6-4564-b31a-c701c0226fbc	+	2e92fbd0-4577-405f-a0c4-5123179a9e78	+	8	0.7840616303663797
```

Column descriptions:
- `query_name`: The name of the query sequence
- `query_orientation`: The orientation of the query sequence (`+` for forward, `-` for reverse complement)
- `target_name`: The name of the target sequence
- `target_orientation`: The orientation of the target sequence (`+` for forward, `-` for reverse complement)
- `neighbor_rank`: The similarity rank of the target sequence among all potential matches for the query sequence, where `1` is the best match, `2` is the second best, and so on.
- `distance`: Measures the dissimilarity between the embedded vectors of the query and target sequences. A smaller value indicates higher similarity between the sequences.



## Program Flowchart

The following flowchart illustrates the main steps of the FEDRANN pipeline. The pipeline implements four operational steps that realize the three core algorithmic concepts (feature extraction encompasses steps 1-3, with dimensionality reduction precomputed in step 2 and applied in step 3):

```
                                    INPUT
                                      |
                                      v
                    +----------------------------------+
                    |  FASTQ/FASTA Sequencing Reads   |
                    |  (e.g., reads.fasta.gz)         |
                    +----------------------------------+
                                      |
                                      v
        ===============================================================
        |          STEP 1: K-mer Counting and Sampling              |
        ===============================================================
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
        +----------------------+           +-------------------------+
        |  Jellyfish Count     |           |  Sample k-mers          |
        |  - Count all k-mers  |  -------> |  - Sample fraction      |
        |  - Filter by min     |           |  - Generate library     |
        |    multiplicity      |           |  - Forward + Reverse    |
        +----------------------+           +-------------------------+
                                                        |
                                                        v
                                          +-------------------------+
                                          |  kmer_searcher          |
                                          |  - Find k-mer positions |
                                          |  - Build sparse vectors |
                                          +-------------------------+
                                                        |
                                                        v
        ===============================================================
        |   STEP 2: Dimensionality Reduction Matrix Precomputation  |
        ===============================================================
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
        +----------------------+           +-------------------------+
        |  Compute IDF Weights |           |  Generate Random        |
        |  - Calculate inverse |           |  Projection Matrix      |
        |    document freq     |  -------> |  - Sparse Random        |
        |  - Weight k-mers by  |           |    Projection (SRP)     |
        |    informativeness   |           |  - n_components × n_feat|
        +----------------------+           +-------------------------+
                                                        |
                                                        v
                                          +-------------------------+
                                          |  Precompute Matrix      |
                                          |  (SRP_transpose × ICF)  |
                                          +-------------------------+
                                                        |
                                                        v
        ===============================================================
        |          STEP 3: Feature Matrix Generation                |
        ===============================================================
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
        +----------------------+           +-------------------------+
        |  Parse k-mer Indices |           |  Apply Precompute       |
        |  - Read binary file  |  -------> |  Matrix                 |
        |  - Build sparse row  |           |  - Matrix multiplication|
        |  - Both strands      |           |  - Get embeddings       |
        +----------------------+           +-------------------------+
                                                        |
                                                        v
                                          +-------------------------+
                                          |  Embedding Matrix       |
                                          |  (n_reads × n_dim)      |
                                          |  [Optional: Save]       |
                                          +-------------------------+
                                                        |
                                                        v
        ===============================================================
        |     STEP 4: Approximate Nearest Neighbor Search           |
        ===============================================================
                          |
                          v
                +-----------------------------+
                |  NNDescent Algorithm        |
                |  - Build k-NN graph index   |
                |  - Graph-based search       |
                |  - Cosine distance metric   |
                +-----------------------------+
                          |
                          v
                +-----------------------------+
                |  Neighbor Indices &         |
                |  Distance Matrix            |
                +-----------------------------+
                          |
                          v
        ===============================================================
        |                        OUTPUT FILES                         |
        ===============================================================
                                       |
              +------------------------+------------------------+
              |                                                 |
              v                                                 v
    +---------------------+                          +---------------------+
    | overlaps.tsv        |                          | feature_matrix.npz  |
    |                     |                          | (optional)          |
    | - query_name        |                          |                     |
    | - query_orientation |                          | - Embedding vectors |
    | - target_name       |                          | - Sparse format     |
    | - target_orientation|                          |                     |
    | - neighbor_rank     |                          +---------------------+
    | - distance          |
    +---------------------+
```

### Key Implementation Details

1. **K-mer Feature Extraction**:
   - Uses Jellyfish for efficient k-mer counting
   - Filters k-mers by minimum multiplicity to reduce noise from sequencing errors
   - Samples a fraction of k-mers to reduce dimensionality while preserving signal
   - Custom C++ tool (`kmer_searcher`) for fast k-mer position finding

2. **Dimensionality Reduction**:
   - **Sparse Random Projection (SRP)**: Projects high-dimensional k-mer space to lower dimensions
   - **IDF Weighting**: Applies inverse collection frequency (ICF) as a practical approximation of IDF, leveraging Jellyfish's efficient k-mer counting
   - Precomputes combined transformation matrix for efficient batch processing

3. **Feature Matrix Generation**:
   - Processes both forward and reverse complement strands
   - Sparse matrix operations for memory efficiency
   - Outputs normalized embedding vectors for each read

4. **Approximate Nearest Neighbor Search**:
   - **NNDescent**: Graph-based approximate k-NN algorithm
   - Uses cosine distance metric for similarity measurement



## License

License

This project is licensed under the GPL-3.0 License. However, the test dataset located at test/data/reads.fasta.gz is explicitly released under the CC0 1.0 Universal license to facilitate maximum reproducibility.

## Contact

Authors: Jia-Yuan Zhang, Changjiu Miao  
Email: zhangjiayuan@genomics.cn, miaochangjiu@genomics.cn

## Citation

Zhang, J. Y., Miao, C., Qiu, T., Xia, X., He, L., He, J., ... & Dong, Y. (2025). FEDRANN: effective long-read overlap detection based on dimensionality reduction and approximate nearest neighbors. bioRxiv, 2025-05. https://www.biorxiv.org/content/10.1101/2025.05.30.656979v3





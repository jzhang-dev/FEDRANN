# SeqNeighbor
Using the nearest neighbor search algorithm to help solve the similarity sequence search problem.
## Install
```
git clone git@github.com:CJ-Miao/SeqNeighbor.git
cd SeqNeighbor
conda env create -f environment.yaml  -p SeqNeighbor_env
conda activate ./SeqNeighbor
pip install -ve .
```
## Usage
```
SeqNeighbor <command> [option] [arguments ...]
```
SeqNeighbor is equipped with two primary commands: `encode` and `qvt`.
### encode
`encode` command are used to encode the input FASTA file into a numerical matrix using K-mer frequencies.
```
usage: SeqNeighbor encode [-h] -q QUERY -t TARGET [-o OUTPUT_DIR] [-p PREPROCESS] [-k KMER_SIZE] [--kmer-sample-fraction KMER_SAMPLE_FRACTION]
                          [--kmer-sample-seed KMER_SAMPLE_SEED] [--kmer-min-multiplicity KMER_MIN_MULTIPLICITY]
options:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        Path to the query (gzip) FASTA file. (default: None)
  -t TARGET, --target TARGET
                        Path to the target (gzip) FASTA file. (default: None)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save output files. (default: ./encode_dir)
  -p PREPROCESS, --preprocess PREPROCESS
                        Preprocess method you want to implement to matrix.(TF/IDF/TF-IDF/None) (default: IDF)
  -k KMER_SIZE, --kmer-size KMER_SIZE
                        K-mer size. (default: 16)
  --kmer-sample-fraction KMER_SAMPLE_FRACTION
                        Percentage of k-mer used to build feature matrix. (default: 0.005)
  --kmer-sample-seed KMER_SAMPLE_SEED
                        Seed used for randomly sampling k-mers. (default: 356115)
  --kmer-min-multiplicity KMER_MIN_MULTIPLICITY
                        Minimal value for k-mer's frequency using as a feature. (default: 2)
```
### qvt
`qvt` command are used to finding neighbors of query sequences in target sequences.
```
usage: SeqNeighbor qvt [-h] -i ENCODE_DIR -o OUTPUT_FILE [-d DIMENSION_REDUCTION_METHOD] [-n N_DIMENSIONS] [--dm-config DM_CONFIG] [--knn-method KNN_METHOD]
                       [--n-neighbors N_NEIGHBORS] [--knn-config KNN_CONFIG]

options:
  -h, --help            show this help message and exit
  -i ENCODE_DIR, --encode-dir ENCODE_DIR
                        Path to encode directory. (default: None)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Path to neighbor information output. (default: None)
  -d DIMENSION_REDUCTION_METHOD, --dimension-reduction-method DIMENSION_REDUCTION_METHOD
                        Dimension reduction method you want to implement to matrix.(GaussianRP/SparseRP/scBiMap/Spectural/PCA/UMAP/None) (default: GaussianRP)
  -n N_DIMENSIONS, --n-dimensions N_DIMENSIONS
                        Dimension of feature vector after dimensionality reduction (default: 500)
  --dm-config DM_CONFIG
                        Path to the JSON configuration file for specifying dimensionality reduction parameters. (default: None)
  --knn-method KNN_METHOD
                        KNN method you want to find neighbots.(ExactNearestNeighbors/NNDescent/HNSW/MinHash/PQ/SimHash) (default: HNSW)
  --n-neighbors N_NEIGHBORS
                        Nearest neighbor number. (default: 20)
  --knn-config KNN_CONFIG
                        Path to the JSON configuration file for specifying knn parameters. (default: None)
```

## Example 
```
SeqNeighbor encode -q query.fasta -t target.fasta -o encode_dir/ -p IDF -k 21
SeqNeighbor qvt -i encode_dir/ -o query_target_neighbor.tsv -d GaussianRP -n 500 --knn-method HNSW --n-neighbors 15
```
If you intend to perform similarity searches within a single FASTA file, you can simply set both the -q (query) and -t (target) parameters to the same file. SeqNeighbor will detect this and employ a more efficient method to accomplish the task.
```
SeqNeighbor encode -q target.fasta -t target.fasta -o encode_dir/ -p IDF -k 21
SeqNeighbor qvt -i encode_dir/ -o query_target_neighbor.tsv -d GaussianRP -n 500 --knn-method HNSW --n-neighbors 15
```
## Output
### encode command
1. If your input query file and target file are not same, output directory will includ following files:

- `target_feature_matrix.npz` and `query_feature_matrix.npz`

These files contain encoded numerical features from fasta file.

- `query_name.txt` and `target_name.txt`

These files saves read names as lists, because feature matrix only contain sequence's index but not sequence's names.

2. If your input query file and target file are same, output directory will includ following files:

- `feature_matrix.npz`

- `read_name.txt`

SeqNeighbor will detect this is an all-versus-all task and avoid duplicate calculations.

> [!TIP]
> If you wish to **explore alternative encoding methods**, you can manually encode the sequences and save the resulting matrix and the sequence ID list using the above specified filenames. Making the directory input of qvt command, now you can execute `qvt` kNN search utilizing your custom encoding approach.


### qvt command
The qvt command only output one tsv file contain neighbor information, including read names, similarity metric and neighbor rank.

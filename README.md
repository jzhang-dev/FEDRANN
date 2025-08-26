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
  --input <input.fasta|input.fastq> \
  --output-dir <output_directory> \
  --preprocess IDF \
  --kmer-size 16 \
  --threads 32 \
  --dimension-reduction SRP \
  --embedding-dimension 500 \
  --knn NNDescent
```

See all options with:

```bash
fedrann --help
```

## Output

- `metadata.tsv`: Metadata for each read
- `feature_matrix.npz`: (optional) Feature matrix in sparse format
- `overlaps.tsv`: Nearest neighbor pairs and distances
- `fedrann.log`: Log file with detailed pipeline progress

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Contact

Authors: Jia-Yuan Zhang, Changjiu Miao  
Email: zhangjiayuan@genomics.cn, miaochangjiu@genomics.cn

## Citation

Zhang, J. Y., Miao, C., Qiu, T., Xia, X., He, L., He, J., ... & Dong, Y. (2025). FEDRANN: effective long-read overlap detection based on dimensionality reduction and approximate nearest neighbors. bioRxiv, 2025-05. https://www.biorxiv.org/content/10.1101/2025.05.30.656979v3





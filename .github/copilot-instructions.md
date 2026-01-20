# Copilot Instructions for FEDRANN

## Project Overview
FEDRANN is a scalable bioinformatics pipeline for overlap detection in large-scale sequencing data. The pipeline uses three main steps: **f**eature **e**xtraction, **d**imensionality **r**eduction, and **a**pproximate **k**-**n**earest **n**eighbor search.

## Technology Stack
- **Language**: Python 3.12
- **Package Management**: Conda (environment.yml) and pip (requirements.txt)
- **Key Dependencies**:
  - BioPython for sequence handling
  - NumPy 1.26.4 (pinned)
  - scikit-learn 1.5.0 for machine learning
  - hnswlib 0.7.0 for approximate nearest neighbors
  - pynndescent 0.5.12 for approximate nearest neighbors
  - numba for JIT compilation
  - scipy for sparse matrices
- **External Components**: C++ kmer_searcher submodule (requires compilation)

## Build and Installation
```bash
# Set up Conda environment
conda env create -p ./env -f environment.yml

# Build kmer_searcher (C++ submodule)
cd kmer_searcher && ./build.sh && cd ..

# Install package
pip install .
```

## Testing
```bash
# Run the test suite
test/test.sh
```
The test script runs the full pipeline with sample data in `test/data/` and outputs to `test/output/`.

## Code Style and Conventions
- **Type Hints**: Always use Python type hints for function parameters and return values. Use types from `typing` module (e.g., `Optional`, `Sequence`, `Mapping`, `Literal`, `NDArray`).
- **Imports**: Import from `numpy.typing` for `NDArray` type hints.
- **Sparse Matrices**: Use `scipy.sparse.csr_matrix` for sparse matrix operations.
- **Logging**: Use the custom logger from `fedrann.custom_logging` module, not print statements.
- **File I/O**: Support both FASTA and FASTQ formats (compressed and uncompressed).
- **Numba**: Use `@njit` decorator for performance-critical numerical operations.

## Architecture
- **Entry Point**: `fedrann/__main__.py` contains the main CLI interface and pipeline orchestration.
- **Feature Extraction**: `feature_extraction.py` handles k-mer counting and feature matrix generation.
- **Dimensionality Reduction**: `dim_reduction.py` implements various DR algorithms (e.g., mpsrp).
- **Nearest Neighbors**: `nearest_neighbors.py` implements exact and approximate kNN search (ExactNearestNeighbors, NNDescent_ava, HNSW).
- **Input Handling**: `fastx_io.py` provides readers for FASTA/FASTQ files.
- **Logging**: `custom_logging.py` provides centralized logging functionality.

## Key Constraints
- **Memory Efficiency**: The pipeline handles large-scale sequencing data. Use sparse matrices where appropriate and be mindful of memory usage.
- **Parallelization**: Support multi-threading via `--threads` parameter. Use thread-safe operations.
- **Reproducibility**: Always respect the `--seed` parameter for random operations to ensure reproducible results.
- **File Format Support**: Maintain compatibility with both compressed (.gz) and uncompressed FASTA/FASTQ files.

## Testing Guidelines
- Test changes using `test/test.sh` which runs a small-scale end-to-end test.
- For CI/CD, the GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push and PR.
- Tests include help message verification and full workflow execution.

## Common Commands
```bash
# Run pipeline with typical parameters
fedrann -i input.fasta.gz -o output/ -k 15 --kmer-sample-fraction 0.05 --seed 602 --kmer-min-multiplicity 2 --dimension-reduction mpsrp --threads 32

# Get help
fedrann --help

# Run via entrypoint script
./entrypoint.sh --help
```

## Dependencies Notes
- Several dependencies have pinned versions (numpy==1.26.4, hnswlib==0.7.0, etc.) - do not change these without testing.
- The project uses both conda and pip dependencies - keep `environment.yml` and `requirements.txt` in sync for overlapping packages.
- External C++ code in `kmer_searcher` submodule must be built before package installation.

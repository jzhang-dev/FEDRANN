#!/usr/bin/env bash

set -exuo pipefail

SeqNeighbor encode \
    -q test/data/reads.fasta.gz \
    -t test/data/reads.fasta.gz \
    -o test/output/feature_extraction \
    -p IDF \
    -k 15 \
    --kmer-sample-fraction 0.05 \
    --kmer-sample-seed 602 \
    --kmer-min-multiplicity 2


SeqNeighbor qvt \
    -i test/output/feature_extraction \
    -d SparseRP \
    -n 500 \
    --knn-method NNDescent \
    -o test/output/KNN \
    --n-neighbors 20 

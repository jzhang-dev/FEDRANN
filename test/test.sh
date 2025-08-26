#!/usr/bin/env bash

set -exuo pipefail

export PATH="$(pwd)/kmer_searcher/build/:$PATH"

rm -rf test/output/
mkdir -p test/output/

./entrypoint.sh \
    -i test/data/reads.fasta.gz \
    -o test/output/ \
    -k 15 \
    --kmer-sample-fraction 0.05 \
    --seed 602 \
    --kmer-min-multiplicity 2 \
    --dimension-reduction mpsrp \
    --save-feature-matrix \
    --threads 2
#!/usr/bin/env bash

set -exuo pipefail

rm -rf test/output/
mkdir -p test/output/

./entrypoint.sh \
    -i test/data/reads.fasta.gz \
    -o test/output/ \
    -k 15 \
    --kmer-sample-fraction 0.05 \
    --seed 602 \
    --kmer-min-multiplicity 2 \
    --threads 2


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
    --threads 2

# mprof plot --output test/output/mprof_plot.png test/output/mprof/memory_profile.dat
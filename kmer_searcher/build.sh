#!/usr/bin/env bash

mkdir -p build
g++ -O3 -march=native -std=c++17 -I ../external/robin-hood-hashing/src/include -o build/kmer_searcher kmer_searcher.cpp -lpthread
export PATH=$(pwd)/build:$PATH
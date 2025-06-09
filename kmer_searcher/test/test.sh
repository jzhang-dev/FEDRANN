#!/usr/bin/env bash

# Test FASTQ
rm -rf test/output/test1
mkdir -p test/output/test1
build/kmer_searcher test/data/test1.kmers.txt test/data/test1.fastq test/output/test1/ 15 1

expected_md5="aba346b3d831a2f3b7f539de5a25cb93"
actual_md5=$(md5sum test/output/test1/output.txt | awk '{ print $1 }')
if [ "$actual_md5" == "$expected_md5" ]; then 
    echo "Test 1 passed: MD5 checksum matches."
else exit 1; fi


# Test FASTA
rm -rf test/output/test2
mkdir -p test/output/test2
build/kmer_searcher test/data/test2.kmers.txt test/data/test2.fasta test/output/test2/ 13 1
expected_md5="91b35034cb21e33ce07189d08f3860b8"
actual_md5=$(md5sum test/output/test2/output.txt | awk '{ print $1 }')
if [ "$actual_md5" == "$expected_md5" ]; then 
    echo "Test 2 passed: MD5 checksum matches."
else exit 1; fi

# Test reading from STDIN
rm -rf test/output/test3
mkdir -p test/output/test3
cat test/data/test1.fastq | build/kmer_searcher test/data/test1.kmers.txt /dev/stdin test/output/test3/ 15 1
expected_md5="aba346b3d831a2f3b7f539de5a25cb93"
actual_md5=$(md5sum test/output/test3/output.txt | awk '{ print $1 }')
if [ "$actual_md5" == "$expected_md5" ]; then 
    echo "Test 3 passed: MD5 checksum matches."
else exit 1; fi

rm -rf test/output/test4
mkdir -p test/output/test4
cat test/data/test1.kmers.txt | build/kmer_searcher /dev/stdin test/data/test1.fastq test/output/test4/ 15 1
expected_md5="aba346b3d831a2f3b7f539de5a25cb93"
actual_md5=$(md5sum test/output/test4/output.txt | awk '{ print $1 }')
if [ "$actual_md5" == "$expected_md5" ]; then 
    echo "Test 4 passed: MD5 checksum matches."
else exit 1; fi





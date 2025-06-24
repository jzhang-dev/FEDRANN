#ifndef KMER_SEARCHER_H
#define KMER_SEARCHER_H

#include <vector>
#include <string>
#include <robin_hood.h>
#include <cstdint>

class KmerSearcher {
public:
    KmerSearcher(int k);
    ~KmerSearcher();

    void load_kmer_libs(const std::string& kmer_lib_path1, const std::string& kmer_lib_path2);
    void process_sequences(const std::string& input_file, int num_threads);
    std::vector<std::pair<std::string, std::vector<uint64_t>>> get_results();

private:
    int K_;
    robin_hood::unordered_map<uint64_t, uint64_t> kmer_to_index_;
    std::vector<std::pair<std::string, std::vector<uint64_t>>> results_;
};

#endif // KMER_SEARCHER_H
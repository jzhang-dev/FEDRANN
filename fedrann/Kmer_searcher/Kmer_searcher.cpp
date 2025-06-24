#include "Kmer_searcher.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <filesystem>
#include <algorithm>
#include <climits>
#include <robin_hood.h>

namespace fs = std::filesystem;

const int BATCH_SIZE = 1000;

struct SequenceRecord {
    std::string id;
    std::string sequence;
};

template <typename T>
class ThreadSafeQueue {
public:
    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
    }

    bool pop_batch(std::vector<T>& batch, int batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;

        while (!queue_.empty() && batch.size() < batch_size) {
            batch.emplace_back(std::move(queue_.front()));
            queue_.pop();
        }
        return !batch.empty();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
};

KmerSearcher::KmerSearcher(int k) : K_(k) {}

KmerSearcher::~KmerSearcher() {}

uint64_t kmer_to_int(const std::string& kmer) {
    uint64_t val = 0;
    for (char c : kmer) {
        val <<= 2;
        switch (c) {
            case 'A': case 'a': break;
            case 'C': case 'c': val |= 1; break;
            case 'G': case 'g': val |= 2; break;
            case 'T': case 't': val |= 3; break;
            default: return UINT64_MAX;
        }
    }
    return val;
}


void KmerSearcher::load_kmer_libs(const std::string& kmer_lib_path1, const std::string& kmer_lib_path2) {
    auto load_single = [&](const std::string& path) {
        std::ifstream kmer_file(path);
        if (!kmer_file) throw std::runtime_error("Cannot open file: " + path);
        
        std::string line;
        uint64_t index = kmer_to_index_.size();
        while (std::getline(kmer_file, line)) {
            if (line.empty() || line[0] == '>') continue;
            if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                line.pop_back();
            }
            if (line.size() != K_) continue;
            
            uint64_t code = kmer_to_int(line);
            if (code != UINT64_MAX && kmer_to_index_.find(code) == kmer_to_index_.end()) {
                kmer_to_index_[code] = index++;
            }
        }
    };

    load_single(kmer_lib_path1);
    load_single(kmer_lib_path2);
}


void read_sequences(const std::string& filename, ThreadSafeQueue<SequenceRecord>& queue) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    bool is_fastq = false;
    bool checked_format = false;
    std::string current_id, current_seq;
    while (std::getline(file, line)) {
        if (!checked_format) {
            is_fastq = (line[0] == '@');
            checked_format = true;
        }

        if (line.empty()) continue;

        if (!is_fastq) {
            if (line[0] == '>') {
                if (!current_id.empty()) {
                    queue.push(SequenceRecord{current_id, current_seq});
                }
                
                size_t first_space = line.find_first_of(" \t");
                if (first_space != std::string::npos) {
                    current_id = line.substr(1, first_space - 1);
                } else {
                    current_id = line.substr(1);
                }
                current_seq.clear();
            } else {
                current_seq += line;
            }
        } else {
            if (line[0] == '@') {
                if (!current_id.empty()) {
                    queue.push(SequenceRecord{current_id, current_seq});
                }
                current_id = line.substr(1);
                std::getline(file, current_seq);
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    if (!is_fastq && !current_id.empty()) {
        queue.push(SequenceRecord{current_id, current_seq});
    }
}


void KmerSearcher::process_sequences(const std::string& input_file, int num_threads) {
    // 先读取所有序列到内存
    std::vector<SequenceRecord> all_sequences;
    {
        ThreadSafeQueue<SequenceRecord> queue;
        std::thread reader([&]() {
            read_sequences(input_file, queue);
        });
        reader.join();
        
        std::vector<SequenceRecord> batch;
        while (queue.pop_batch(batch, INT_MAX)) {
            all_sequences.insert(all_sequences.end(), batch.begin(), batch.end());
        }
    }
    
    std::cout << "Read " << all_sequences.size() << " sequences" << std::endl;
    
    // 并行处理序列
    results_.resize(all_sequences.size());
    std::mutex results_mutex;
    
    auto process_chunk = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const auto& record = all_sequences[i];

            robin_hood::unordered_set<uint64_t> index_set;
         
            uint64_t current_kmer = 0;
            uint64_t mask = (1ULL << (2*K_)) - 1;

            // 初始化第一个k-mer
            for (int i = 0; i < K_ && i < record.sequence.size(); ++i) {
                current_kmer = (current_kmer << 2) & mask;
                switch (record.sequence[i]) {
                    case 'A': case 'a': break;
                    case 'C': case 'c': current_kmer |= 1; break;
                    case 'G': case 'g': current_kmer |= 2; break;
                    case 'T': case 't': current_kmer |= 3; break;
                    default: current_kmer = UINT64_MAX; break;
                }
            }

            if (current_kmer != UINT64_MAX && kmer_to_index_.count(current_kmer)) {
                index_set.insert(kmer_to_index_[current_kmer]);
            }

            // 滑动窗口
            for (int i = K_; i < record.sequence.size(); ++i) {
                current_kmer = (current_kmer << 2) & mask;
                switch (record.sequence[i]) {
                    case 'A': case 'a': break;
                    case 'C': case 'c': current_kmer |= 1; break;
                    case 'G': case 'g': current_kmer |= 2; break;
                    case 'T': case 't': current_kmer |= 3; break;
                    default: current_kmer = UINT64_MAX; break;
                }

                if (current_kmer != UINT64_MAX && kmer_to_index_.count(current_kmer)) {
                    index_set.insert(kmer_to_index_[current_kmer]);
                }
            }

            std::lock_guard<std::mutex> lock(results_mutex);
            results_[i] = {record.id, 
                          std::vector<uint64_t>(index_set.begin(), index_set.end())};
        }
    };
    
    std::vector<std::thread> threads;
    int chunk_size = all_sequences.size() / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? all_sequences.size() : (i + 1) * chunk_size;
        threads.emplace_back(process_chunk, start, end);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}


std::vector<std::pair<std::string, std::vector<uint64_t>>> KmerSearcher::get_results() {
    return std::move(results_);
}
#ifndef KMER_SEARCHER_H
#define KMER_SEARCHER_H

#include <vector>
#include <string>
#include <robin_hood.h>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <queue>
#include <memory>
#include <functional>
#include <fstream>
#include <iostream>
#include <iomanip>

class KmerSearcher {
public:
    explicit KmerSearcher(size_t k = 31);
    ~KmerSearcher();
    
    void load_kmer_libs(const std::string& kmer_lib_path1, const std::string& kmer_lib_path2);
    void process_sequences(const std::string& input_file, const std::string& output_seqid_file, int num_threads);
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> get_results();

    const uint64_t* get_col_indices_ptr() const;
    size_t get_col_indices_size() const;
    const uint64_t* get_indptr_ptr() const;
    size_t get_indptr_size() const;

    KmerSearcher(const KmerSearcher&) = delete;
    KmerSearcher& operator=(const KmerSearcher&) = delete;

private:
    size_t K_;
    size_t kmer_count_; // 存储第一个kmer库的大小
    robin_hood::unordered_map<uint64_t, uint64_t> kmer_to_index_;
    std::ofstream output_file_; // 输出序列ID的文件
    
    // 序列记录结构
    struct SequenceRecord {
        uint64_t global_index; // 全局序列索引
        std::string id;
        std::string sequence;
        
        SequenceRecord() = default;
        SequenceRecord(uint64_t index, std::string id, std::string seq) 
            : global_index(index), id(std::move(id)), sequence(std::move(seq)) {}
        
        SequenceRecord(SequenceRecord&& other) noexcept
            : global_index(other.global_index), 
              id(std::move(other.id)), 
              sequence(std::move(other.sequence)) {}
        
        SequenceRecord& operator=(SequenceRecord&& other) noexcept {
            global_index = other.global_index;
            id = std::move(other.id);
            sequence = std::move(other.sequence);
            return *this;
        }
        
        SequenceRecord(const SequenceRecord&) = delete;
        SequenceRecord& operator=(const SequenceRecord&) = delete;
    };
    
    // 线程安全队列
    template <typename T>
    class ThreadSafeQueue {
    public:
        void push(T&& value) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }

        bool pop_batch(std::vector<T>& batch, size_t batch_size) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return false;

            batch.reserve(batch_size);
            while (!queue_.empty() && batch.size() < batch_size) {
                batch.emplace_back(std::move(queue_.front()));
                queue_.pop();
            }
            return true;
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
    };
    
    // 内存映射文件类
    class MemoryMappedFile {
    public:
        explicit MemoryMappedFile(const std::string& filepath);
        ~MemoryMappedFile();
        
        const char* data() const { return data_; }
        size_t size() const { return size_; }
        
        MemoryMappedFile(const MemoryMappedFile&) = delete;
        MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    private:
        const char* data_;
        size_t size_;
        int fd_;
    };
    
    // 私有方法
    static uint64_t kmer_to_int(const std::string& kmer);
    void read_sequences_fast(const std::string& filename, ThreadSafeQueue<SequenceRecord>& queue);
    void log_timestamp(const std::string& message) const;
    
    // 同步原语
    std::mutex results_mutex_;
    std::vector<uint64_t> col_indices_; // 存储所有kmer索引
    std::vector<uint64_t> indptr_;      // 存储每个read的边界位置
    std::mutex file_mutex_;             // 文件输出锁
    std::atomic<uint64_t> global_seq_index_{0}; // 全局序列计数器
};

#endif // KMER_SEARCHER_H
#include "Kmer_searcher.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cctype>
#include <algorithm>


constexpr size_t BATCH_SIZE = 1000;

void KmerSearcher::log_timestamp(const std::string& message) const {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time);
    
    std::cout << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] " 
              << message << std::endl;
}

KmerSearcher::MemoryMappedFile::MemoryMappedFile(const std::string& filepath) 
    : data_(nullptr), size_(0), fd_(-1) {
    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("fstat failed for file: " + filepath);
    }
    
    size_ = sb.st_size;
    if (size_ > 0) {
        data_ = static_cast<const char*>(
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0)
        );
        if (data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed for file: " + filepath);
        }
    }
}

KmerSearcher::MemoryMappedFile::~MemoryMappedFile() {
    if (data_ != nullptr && data_ != MAP_FAILED && size_ > 0) {
        munmap(const_cast<char*>(data_), size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

KmerSearcher::KmerSearcher(size_t k) : K_(k), kmer_count_(0) {
    indptr_.push_back(0); // 初始化indptr
}

KmerSearcher::~KmerSearcher() {
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

uint64_t KmerSearcher::kmer_to_int(const std::string& kmer) {
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
        log_timestamp("Loading kmer lib: " + path);
        
        std::ifstream kmer_file(path);
        if (!kmer_file) throw std::runtime_error("Cannot open file: " + path);
        
        std::string line;
        uint64_t index = kmer_to_index_.size();
        size_t count = 0;
        
        while (std::getline(kmer_file, line)) {
            if (line.empty() || line[0] == '>') continue;
            if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                line.pop_back();
            }
            if (line.size() != K_) continue;
            uint64_t code = kmer_to_int(line);
            if (code != UINT64_MAX && kmer_to_index_.find(code) == kmer_to_index_.end()) {
                kmer_to_index_[code] = index++;
                count++;
            }
        }
        
        log_timestamp("Loaded " + std::to_string(count) + " kmers from " + path);
        return count;
    };

    size_t count1 = load_single(kmer_lib_path1);
    kmer_count_ = count1; // 记录第一个库的大小
    size_t count2 = load_single(kmer_lib_path2);
    log_timestamp("Total kmer index size: " + std::to_string(kmer_to_index_.size()));
}

void KmerSearcher::read_sequences_fast(const std::string& filename, ThreadSafeQueue<SequenceRecord>& queue) {
    log_timestamp("Reading sequences from: " + filename);
    
    try {
        MemoryMappedFile mmap_file(filename);
        const char* data = mmap_file.data();
        size_t size = mmap_file.size();
        
        if (size == 0) {
            log_timestamp("Empty file: " + filename);
            return;
        }
        
        const char* end = data + size;
        const char* ptr = data;
        
        std::vector<SequenceRecord> batch;
        batch.reserve(BATCH_SIZE);
        
        SequenceRecord current_record;
        bool in_header = false;
        bool in_sequence = false;
        bool is_fastq = false;
        bool checked_format =false;
        bool in_quality = false;
        
        while (ptr < end) {
            if (!checked_format) {
                is_fastq = (*ptr == '@');
                checked_format = true;
            }
            
            if (*ptr == '\n') {
                ptr++;
                continue;
            }
            
            if (!in_header && !in_sequence && !in_quality) {
                if (*ptr == '>' || (is_fastq && *ptr == '@')) {
                    in_header = true;
                    uint64_t index = global_seq_index_++;
                    current_record = SequenceRecord(index, "", "");
                    ptr++;
                }
            }
            
            if (in_header) {
                const char* id_start = ptr;
                while (ptr < end && *ptr != '\n' && *ptr != ' ' && *ptr != '\t') {
                    ptr++;
                }
                
                current_record.id = std::string(id_start, ptr - id_start);
                in_header = false;
                
                if (!is_fastq) {
                    in_sequence = true;
                }
                
                while (ptr < end && *ptr != '\n') ptr++;
                if (ptr < end) ptr++;
            }
            
            if (in_sequence) {
                const char* seq_start = ptr;
                while (ptr < end && *ptr != '>' && *ptr != '@' && *ptr != '+') {
                    if (*ptr == '\n') {
                        ptr++;
                        continue;
                    }
                    ptr++;
                }
                
                std::string sequence(seq_start, ptr - seq_start);
                for (char& c : sequence) {
                    if (c >= 'a' && c <= 'z') {
                        c = c - 'a' + 'A';
                    }
                }
                current_record.sequence = sequence;
                batch.push_back(std::move(current_record));
                if (batch.size() >= BATCH_SIZE) {
                    for (auto& record : batch) {
                        queue.push(std::move(record));
                    }
                    batch.clear();
                    batch.reserve(BATCH_SIZE);
                }
                
                in_sequence = false;
                
                if (is_fastq && ptr < end && *ptr == '+') {
                    in_quality = true;
                    while (ptr < end && *ptr != '\n') ptr++;
                    if (ptr < end) ptr++;
                }
            }
            
            if (in_quality) {
                while (ptr < end && *ptr != '\n') ptr++;
                if (ptr < end) ptr++;
                in_quality = false;
            }
        }
        
        if (!batch.empty()) {
            for (auto& record : batch) {
                queue.push(std::move(record));
            }
        }
    } catch (const std::exception& e) {
        log_timestamp("Memory mapping failed: " + std::string(e.what()));
        
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
                        uint64_t index = global_seq_index_++;
                        queue.push(SequenceRecord{index, std::move(current_id), std::move(current_seq)});
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
                        uint64_t index = global_seq_index_++;
                        queue.push(SequenceRecord{index, std::move(current_id), std::move(current_seq)});
                    }
                    current_id = line.substr(1);
                    std::getline(file, current_seq);
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
        }

        if (!is_fastq && !current_id.empty()) {
            uint64_t index = global_seq_index_++;
            queue.push(SequenceRecord{index, std::move(current_id), std::move(current_seq)});
        }
    }
    
    log_timestamp("Finished reading sequences from: " + filename);
}

void KmerSearcher::process_sequences(const std::string& input_file, 
                                   const std::string& output_seqid_file, 
                                   int num_threads) {
    log_timestamp("Starting sequence processing with " + std::to_string(num_threads) + " threads");
    
    // 重置全局序列计数器
    global_seq_index_ = 0;
    col_indices_.clear();
    indptr_.clear();
    indptr_.push_back(0); // 初始化为0
    
    // 打开输出文件
    output_file_.open(output_seqid_file);
    if (!output_file_.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_seqid_file);
    }
    output_file_ << "index" << "\t" << "read_name" << "\tstrand\n";
    ThreadSafeQueue<SequenceRecord> work_queue;
    std::atomic<size_t> total_sequences{0};
    std::atomic<bool> reading_completed{false};
    std::atomic<bool> error_occurred{false};
    
    // 启动生产者线程
    std::thread producer([&]() {
        try {
            read_sequences_fast(input_file, work_queue);
        } catch (const std::exception& e) {
            log_timestamp("Error reading sequences: " + std::string(e.what()));
            error_occurred = true;
        }
        reading_completed = true;
    });
    
    // 消费者线程函数
    auto consumer = [&](size_t thread_id) {
        std::vector<SequenceRecord> batch;
        batch.reserve(BATCH_SIZE);
        
        // 线程本地存储
        struct ThreadResult {
            uint64_t global_index;  // 该序列的全局索引
            std::string read_name;  // 序列ID
            std::vector<uint64_t> forward_indices;  // 正向kmer索引
            std::vector<uint64_t> col_indices;      // 正向+反向kmer索引
            std::vector<uint64_t> indptr;           // 本地indptr
        };
        
        std::vector<ThreadResult> thread_results;
        thread_results.reserve(BATCH_SIZE);
        
        while (!error_occurred) {
            batch.clear();
            if (!work_queue.pop_batch(batch, BATCH_SIZE)) {
                if (reading_completed) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            for (auto& record : batch) {
                const std::string& seq = record.sequence;
                const size_t seq_len = seq.size();
                ThreadResult result;
                result.global_index = record.global_index;
                result.read_name = record.id;
                
                // 提取正向kmer索引
                if (seq_len >= K_) {
                    uint64_t current_kmer = 0;
                    const uint64_t mask = (1ULL << (2 * K_)) - 1;
                    
                    // 初始化第一个k-mer
                    for (size_t i = 0; i < K_; i++) {
                        current_kmer <<= 2;
                        switch (seq[i]) {
                            case 'A': case 'a': break;
                            case 'C': case 'c': current_kmer |= 1; break;
                            case 'G': case 'g': current_kmer |= 2; break;
                            case 'T': case 't': current_kmer |= 3; break;
                            default: current_kmer = UINT64_MAX; break;
                        }
                    }
                    current_kmer &= mask;
                    
                    if (current_kmer != UINT64_MAX) {
                        auto it = kmer_to_index_.find(current_kmer);
                        if (it != kmer_to_index_.end()) {
                            result.forward_indices.push_back(it->second);
                        }
                    }
                    
                    // 滑动窗口处理剩余序列
                    for (size_t i = K_; i < seq_len; i++) {
                        current_kmer = (current_kmer << 2) & mask;
                        switch (seq[i]) {
                            case 'A': case 'a': break;
                            case 'C': case 'c': current_kmer |= 1; break;
                            case 'G': case 'g': current_kmer |= 2; break;
                            case 'T': case 't': current_kmer |= 3; break;
                            default: current_kmer = UINT64_MAX; break;
                        }
                        
                        if (current_kmer != UINT64_MAX) {
                            auto it = kmer_to_index_.find(current_kmer);
                            if (it != kmer_to_index_.end()) {
                                result.forward_indices.push_back(it->second);
                            }
                        }
                    }
                }
                
                // 添加正向kmer索引
                result.col_indices = result.forward_indices;
                
                // 添加反向kmer索引
                for (uint64_t idx : result.forward_indices) {
                    // 计算反向索引：如果索引在第一个库中，则加上kmer_count，否则减去kmer_count
                    uint64_t rev_idx = (idx < kmer_count_) ? (idx + kmer_count_) : (idx - kmer_count_);
                    result.col_indices.push_back(rev_idx);
                }
                
                // 设置本地indptr
                size_t forward_end = result.forward_indices.size();
                size_t total_end = result.col_indices.size();
                result.indptr.push_back(forward_end);
                result.indptr.push_back(total_end);
                
                thread_results.push_back(std::move(result));
                total_sequences++;
            }
            
            // 批量更新全局结果
            if (!thread_results.empty()) {
                std::lock_guard<std::mutex> lock1(results_mutex_);
                std::lock_guard<std::mutex> lock2(file_mutex_);
                
                // 按全局索引排序，确保输出顺序与输入顺序一致
                std::sort(thread_results.begin(), thread_results.end(),
                         [](const ThreadResult& a, const ThreadResult& b) {
                             return a.global_index < b.global_index;
                         });
                
                // 处理每个结果
                for (auto& res : thread_results) {
                    // 计算全局偏移量
                    uint64_t col_base = col_indices_.size();
                    uint64_t indptr_base = indptr_.back();
                    
                    // 更新col_indices_
                    col_indices_.insert(col_indices_.end(), 
                                      res.col_indices.begin(), 
                                      res.col_indices.end());
                    
                    // 更新indptr_
                    for (uint64_t pos : res.indptr) {
                        indptr_.push_back(indptr_base + pos);
                    }
                    
                    // 输出序列ID（两行）
                    uint64_t base_line = res.global_index * 2;
                    output_file_ << base_line << "\t" << res.read_name << "\t0\n";
                    output_file_ << base_line + 1 << "\t" << res.read_name << "\t1\n";
                }
                
                thread_results.clear();
            }
        }
        
        // 处理剩余线程本地结果
        if (!thread_results.empty()) {
            std::lock_guard<std::mutex> lock1(results_mutex_);
            std::lock_guard<std::mutex> lock2(file_mutex_);
            
            // 按全局索引排序
            std::sort(thread_results.begin(), thread_results.end(),
                     [](const ThreadResult& a, const ThreadResult& b) {
                         return a.global_index < b.global_index;
                     });
            
            // 处理每个结果
            for (auto& res : thread_results) {
                uint64_t col_base = col_indices_.size();
                uint64_t indptr_base = indptr_.back();
                
                col_indices_.insert(col_indices_.end(), 
                                  res.col_indices.begin(), 
                                  res.col_indices.end());
                
                for (uint64_t pos : res.indptr) {
                    indptr_.push_back(indptr_base + pos);
                }
                
                uint64_t base_line = res.global_index * 2;
                output_file_ << base_line << "\t" << res.read_name << "\t0\n";
                output_file_ << base_line + 1 << "\t" << res.read_name << "\t1\n";
            }
        }
    };
    
    // 启动消费者线程
    std::vector<std::thread> consumers;
    for (int i = 0; i < num_threads; ++i) {
        consumers.emplace_back(consumer, i);
    }
    
    // 等待生产者完成
    producer.join();
    
    // 等待消费者完成
    for (auto& t : consumers) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // 确保文件刷新
    output_file_.flush();
    
    log_timestamp("Processed " + std::to_string(total_sequences.load()) + " sequences");
    log_timestamp("Total kmer indices: " + std::to_string(col_indices_.size()));
    log_timestamp("Indptr size: " + std::to_string(indptr_.size()));
}

const uint64_t* KmerSearcher::get_col_indices_ptr() const {
    return col_indices_.data();
}

size_t KmerSearcher::get_col_indices_size() const {
    return col_indices_.size();
}

const uint64_t* KmerSearcher::get_indptr_ptr() const {
    return indptr_.data();
}

size_t KmerSearcher::get_indptr_size() const {
    return indptr_.size();
}

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> KmerSearcher::get_results() {
    // 确保indptr包含起始0
    if (indptr_.empty()) {
        indptr_.push_back(0);
    }
    
    return {std::move(col_indices_), std::move(indptr_)};
}
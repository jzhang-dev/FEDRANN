#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <filesystem>
#include <robin_hood.h>
#include <map>
#include <unordered_set>
namespace fs = std::filesystem;

const int K = 31; // k-mer长度
const int NUM_THREADS = 64; // 改为64线程
const int BATCH_SIZE = 1000; // 批次大小

// 序列记录结构体
struct SequenceRecord {
    std::string id;
    std::string sequence;
};

// 线程安全队列
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

using LocalKmerStats = std::unordered_map<uint64_t, uint64_t>; 

class ThreadSafeKmerStats {
public:
    // 构造函数：初始化向量的大小
    ThreadSafeKmerStats(size_t num_unique_kmers) : counts_(num_unique_kmers, 0) {}

    // 1. 【核心功能】安全地合并本地统计结果
    void merge(const LocalKmerStats& local_counts) {
        // 依然只在这里使用一次锁
        std::lock_guard<std::mutex> lock(mutex_); 
        for (const auto& pair : local_counts) {
            // 注意：这里 pair.first 必须是 k-mer 索引 (0, 1, 2...)
            // 而不是 k-mer 编码！
            // 假设 LocalKmerStats 在主函数中被修改为使用索引作为键
            uint64_t kmer_index = pair.first;
            counts_[kmer_index] += pair.second;
        }
    } 

    // 2. 安全地获取最终结果（返回副本或引用，取决于需求）
    // 返回 const 引用，避免复制整个巨大的向量
    const std::vector<uint64_t>& get_counts() const {
        // 在返回引用前加锁，确保读取时数据一致
        std::lock_guard<std::mutex> lock(mutex_); 
        return counts_; 
    }
    
private:
    // 替换为 vector，其索引就是 k-mer 索引
    std::vector<uint64_t> counts_;
    mutable std::mutex mutex_;
};


// 线程安全的输出存储
class ThreadSafeOutput {
public:
    void add_result(const std::string& id, const robin_hood::unordered_set<uint64_t>& index_set) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<uint64_t> indices(index_set.begin(), index_set.end());
        results_.emplace_back(id, std::move(indices));
    }

    void write_to_file(const std::string& filename) {
        std::ofstream outfile(filename, std::ios::binary);
        
        const char magic[4] = {'K','M','E','R'};
        const uint8_t version = 1;
        const char reserved[3] = {0};
        outfile.write(magic, 4);
        outfile.write(reinterpret_cast<const char*>(&version), 1);
        outfile.write(reserved, 3);
        
        uint64_t total_records = results_.size();
        outfile.write(reinterpret_cast<const char*>(&total_records), 8);
        
        for (const auto& [id, indices] : results_) {
            if (!std::all_of(id.begin(), id.end(), [](char c) {
                return c >= 32 && c <= 126;
            })) {
                throw std::runtime_error("ID contains non-ASCII characters");
            }
            
            uint16_t id_len = id.size();
            outfile.write(reinterpret_cast<const char*>(&id_len), 2);
            outfile.write(id.data(), id_len);
            
            uint32_t index_count = indices.size();
            outfile.write(reinterpret_cast<const char*>(&index_count), 4);
            
            for (const auto& index : indices) {
                uint64_t index_val = index;
                outfile.write(reinterpret_cast<const char*>(&index_val), 8);
            }
        }
    }

private:
    std::vector<std::pair<std::string, std::vector<uint64_t>>> results_;
    mutable std::mutex mutex_;
};

// k-mer转换函数
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

void read_sequences(const std::string& filename, ThreadSafeQueue<SequenceRecord>& queue) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("无法打开文件: " + filename);

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
                current_id = line.substr(1);
                std::getline(file, current_seq);
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                queue.push(SequenceRecord{current_id, current_seq});
            }
        }
    }

    if (!is_fastq && !current_id.empty()) {
        queue.push(SequenceRecord{current_id, current_seq});
    }
}

void writeKmerStatsToBinary(const std::vector<uint64_t>& kmer_record_counts, 
                           const std::string& filename) { 

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // 遍历所有 k-mer。i 就是 k-mer 索引。
    for (size_t i = 0; i < kmer_record_counts.size(); ++i) {
        uint64_t count = kmer_record_counts[i];
        
        // 优化：跳过计数为 0 的 k-mer
        if (count == 0) continue; 
        
        // i 就是我们要存储的 k-mer 索引
        uint64_t kmer_index_val = i; 
        
        // 写入 k-mer 索引
        outfile.write(reinterpret_cast<const char*>(&kmer_index_val), sizeof(kmer_index_val)); 
        // 写入记录数量
        outfile.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }
    
    outfile.close();
}


int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <kmer_lib> <input_fastq> <output_dir> <k> <threads>\n";
        return 1;
    }

    int K = std::stoi(argv[4]);
    if (K <= 0 || K > 31) {
        std::cerr << "Invalid k value: " << argv[4] << std::endl;
        return 1;
    }

    int NUM_THREADS = std::stoi(argv[5]);
    if (NUM_THREADS <= 0) {
        std::cerr << "Invalid number of threads: " << argv[5] << std::endl;
        return 1;
    }

    // 创建输出目录
    std::string output_dir = argv[3];
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }

    // 加载k-mer库
    robin_hood::unordered_map<uint64_t, uint64_t> kmer_to_index;
    std::vector<uint64_t> index_to_kmer;
    {
        std::ifstream kmer_file(argv[1]);
        std::string kmer;
        uint64_t index = 0;
        while (kmer_file >> kmer) {
            if (kmer.size() != K) continue;
            uint64_t code = kmer_to_int(kmer);
            if (code != UINT64_MAX && kmer_to_index.count(code) == 0) {
                kmer_to_index[code] = index++;
                index_to_kmer.push_back(code);
            }
        }
    }
    size_t num_unique_kmers = index_to_kmer.size(); 

    // 实例化 ThreadSafeKmerStats
    ThreadSafeKmerStats global_kmer_stats(num_unique_kmers);
    // 准备多线程处理
    ThreadSafeQueue<SequenceRecord> work_queue;
    std::atomic<bool> done_reading{false};
    ThreadSafeOutput global_output;

    // 生产者线程
    std::thread producer([&]() {
        try {
            read_sequences(argv[2], work_queue);
        } catch (const std::exception& e) {
            std::cerr << "Error reading input: " << e.what() << std::endl;
            exit(1);
        }
        done_reading = true;
    });

    // 消费者线程函数
    auto consumer = [&]() {
        std::vector<SequenceRecord> batch;
        LocalKmerStats local_kmer_stats;
        batch.reserve(BATCH_SIZE);

        while (true) {
            batch.clear();
            if (!work_queue.pop_batch(batch, BATCH_SIZE)) {
                if (done_reading) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            for (const auto& record : batch) {
                robin_hood::unordered_set<uint64_t> index_set;
                uint64_t current_kmer = 0;
                uint64_t mask = (1ULL << (2*K)) - 1;

                // 初始化第一个k-mer
                for (int i = 0; i < K && i < record.sequence.size(); ++i) {
                    current_kmer = (current_kmer << 2) & mask;
                    switch (record.sequence[i]) {
                        case 'A': case 'a': break;
                        case 'C': case 'c': current_kmer |= 1; break;
                        case 'G': case 'g': current_kmer |= 2; break;
                        case 'T': case 't': current_kmer |= 3; break;
                        default: current_kmer = UINT64_MAX; break;
                    }
                }

                if (current_kmer != UINT64_MAX && kmer_to_index.count(current_kmer)) {
                    uint64_t kmer_index = kmer_to_index.at(current_kmer); // 使用 .at() 或 []
                    if (index_set.insert(kmer_index).second) {
                        local_kmer_stats[kmer_index]++; // <-- 更新本地
                        }
                    }

                // 滑动窗口
                for (int i = K; i < record.sequence.size(); ++i) {
                    current_kmer = (current_kmer << 2) & mask;
                    switch (record.sequence[i]) {
                        case 'A': case 'a': break;
                        case 'C': case 'c': current_kmer |= 1; break;
                        case 'G': case 'g': current_kmer |= 2; break;
                        case 'T': case 't': current_kmer |= 3; break;
                        default: current_kmer = UINT64_MAX; break;
                    }

                    if (current_kmer != UINT64_MAX && kmer_to_index.count(current_kmer)) {
                        uint64_t kmer_index = kmer_to_index.at(current_kmer); // 使用 .at() 或 []
                        if (index_set.insert(kmer_index).second) {
                            local_kmer_stats[kmer_index]++; // <-- 更新本地
                            }
                        }
                }
                global_output.add_result(record.id, index_set);
            }
        }
        global_kmer_stats.merge(local_kmer_stats);
    };

    // 启动消费者线程
    std::vector<std::thread> consumers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        consumers.emplace_back(consumer);
    }

    // 等待线程完成
    producer.join();
    for (auto& t : consumers) {
        t.join();
    }

    // 获取最终的统计结果并写入文件
    writeKmerStatsToBinary(global_kmer_stats.get_counts(), fs::path(output_dir) / "kmer_frequency.bin");
    
    // 将所有结果写入最终文件
    global_output.write_to_file(fs::path(output_dir) / "output.bin");

    return 0;
}
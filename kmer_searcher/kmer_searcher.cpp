#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <filesystem>
#include <robin_hood.h>

namespace fs = std::filesystem;

const int K = 31; // k-mer长度
const int NUM_THREADS = 64; // 改为64线程
const int BATCH_SIZE = 1000; // 批次大小

// 序列记录结构体
struct SequenceRecord {
    std::string id;
    std::string sequence;
};

// 线程安全队列（改用阻塞锁 + 批量处理）
template <typename T>
class ThreadSafeQueue {
public:
    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
    }

    // 批量取出数据（阻塞方式）
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

// 线程安全的输出存储
class ThreadSafeOutput {
public:
    void add_result(const std::string& id, const robin_hood::unordered_set<uint64_t>& index_set) {
        std::lock_guard<std::mutex> lock(mutex_);
        // 将 unordered_set 转换为 vector
        std::vector<uint64_t> indices(index_set.begin(), index_set.end());
        results_.emplace_back(id, std::move(indices));
    }

    void write_to_file(const std::string& filename) {
        std::ofstream outfile(filename, std::ios::binary);
        
        // 文件头 (16字节)
        const char magic[4] = {'K','M','E','R'};
        const uint8_t version = 1;
        const char reserved[3] = {0};
        outfile.write(magic, 4);
        outfile.write(reinterpret_cast<const char*>(&version), 1);
        outfile.write(reserved, 3);
        
        // 写入记录总数 (8字节)
        uint64_t total_records = results_.size();
        outfile.write(reinterpret_cast<const char*>(&total_records), 8);
        
        for (const auto& [id, indices] : results_) {
            // 确保ID是ASCII可打印字符
            if (!std::all_of(id.begin(), id.end(), [](char c) {
                return c >= 32 && c <= 126; // ASCII可打印范围
            })) {
                throw std::runtime_error("ID contains non-ASCII characters");
            }
            
            // 写入ID长度(2B)和内容
            uint16_t id_len = id.size();
            outfile.write(reinterpret_cast<const char*>(&id_len), 2);
            outfile.write(id.data(), id_len);
            
            // 写入k-mer索引数(4B)
            uint32_t index_count = indices.size();
            outfile.write(reinterpret_cast<const char*>(&index_count), 4);
            
            // 写入k-mer索引
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
                
                // 修改点：只保留第一个空格/tab前的内容
                size_t first_space = line.find_first_of(" \t");
                if (first_space != std::string::npos) {
                    current_id = line.substr(1, first_space - 1); // 去掉>和后面的空格
                } else {
                    current_id = line.substr(1); // 没有空格则取整行
                }
                
                current_seq.clear();
            } else {
                current_seq += line;
            }
        } else {
            // FASTQ格式处理保持不变
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
                    index_set.insert(kmer_to_index[current_kmer]);
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
                        index_set.insert(kmer_to_index[current_kmer]);
                    }
                }

                // 将结果存入内存
                global_output.add_result(record.id, index_set);
            }
        }
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

    // 将所有结果写入最终文件
    global_output.write_to_file(fs::path(output_dir) / "output.bin");

    return 0;
}
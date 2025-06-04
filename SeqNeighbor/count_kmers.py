import subprocess
import tempfile
import os
from os.path import abspath, isfile, join
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, Optional, Iterable

from .custom_logging import logger
from . import globals


class AsyncJellyfishResult:
    def __init__(self, temp_dir: str, future):
        self._temp_dir = temp_dir
        self._future = future
        self._result_file = os.path.join(temp_dir, "result.txt")

    def done(self) -> bool:
        """检查任务是否完成"""
        return self._future.done()

    def get_result(self) -> Iterable[str]:
        """
        获取结果，如果任务未完成会阻塞直到完成
        返回生成器，产生(kmer, count)元组
        """
        if not self.done():
            self._future.result()  # 等待任务完成
        assert self.done()
        logger.debug(
            "Jellyfish task completed, reading results from %s", self._result_file
        )

        # 读取结果文件
        with open(self._result_file, "r") as f:
            for line in f:
                kmer, count = line.strip().split(" ")
                yield kmer

        self.cleanup()

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self._temp_dir):
            logger.debug("Cleaning up temporary directory: %s", self._temp_dir)
            shutil.rmtree(self._temp_dir)


def run_jellyfish(
    input_file: str,
    k: int,
    min_multiplicity: int,
    threads: int = 1,
    hash_size: str = "10G",
    temp_dir: str | None = None,
) -> AsyncJellyfishResult:
    # 创建临时目录
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="jellyfish_async_")
    os.makedirs(temp_dir, exist_ok=True)
    count_file = os.path.join(temp_dir, "kmer_counts.jf")
    result_file = os.path.join(temp_dir, "result.txt")

    # 定义实际执行的任务函数
    def _run_jellyfish():
        if input_file.endswith(".gz"):
            decompressed_input_file = os.path.join(
                temp_dir, os.path.basename(input_file[:-3])
            )
            unzip_cmd = ["gunzip", "-c", input_file]
            with open(decompressed_input_file, "wb") as out_f:
                logger.debug(f"Unzipping {input_file} to {decompressed_input_file}")
                subprocess.run(unzip_cmd, stdout=out_f, check=True)
        else:
            decompressed_input_file = input_file

        # 执行count命令
        count_cmd = [
            "jellyfish",
            "count",
            "-m",
            str(k),
            "-s",
            hash_size,
            "-t",
            str(threads),
            "-C",
            decompressed_input_file,
            "-o",
            count_file,
        ]
        logger.debug(f"Running Jellyfish count command: {' '.join(count_cmd)}")
        subprocess.run(count_cmd, check=True)
        if not os.path.isfile(count_file):
            raise RuntimeError(f"Jellyfish count output file not found: {count_file}")
        # 执行dump命令并将结果保存到文件
        dump_cmd = ["jellyfish", "dump", "-c", "-L", str(min_multiplicity), count_file]
        with open(result_file, "w") as f:
            logger.debug(f"Running Jellyfish dump command: {' '.join(dump_cmd)}")
            subprocess.run(dump_cmd, stdout=f, check=True)

        return True

    # 使用线程池异步执行
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_jellyfish)

    # 返回结果对象
    return AsyncJellyfishResult(temp_dir, future)


def _parse_kmer_searcher_output(path: str):
    indices = []
    counts = []
    name = ""

    with open(path, "r") as f:
        for line in f:
            line = line.strip("\n")
            if line.startswith(">"):
                if name:
                    yield name, indices, counts
                parts = line[1:].split()
                name = parts[0]
                indices = []
                counts = []
            else:
                index, count = map(int, line.split())
                indices.append(index)
                counts.append(count)
        if name:  # 最后一个记录
            yield name, indices, counts


def get_kmer_features(fasta_path: str, k: int, sample_fraction: float, min_multiplicity: int = 2):
    jf_path = join(globals.temp_dir, "kmer_counts.jf")
    hash_size = "10G"
    threads = globals.threads

    jellyfish_count_command = [
        "jellyfish",
        "count",
        "-m",
        str(k),
        "-s",
        hash_size,
        "-t",
        str(threads),
        "-C",
        fasta_path,
        "-o",
        jf_path,
    ]
    logger.debug(f"Running Jellyfish count command: {' '.join(jellyfish_count_command)}")
    subprocess.run(jellyfish_count_command, check=True)
    if not isfile(jf_path):
        raise RuntimeError(f"Jellyfish count output file not found: {jf_path}")
    

    fwd_kmer_library_path = join(globals.temp_dir, "fwd_kmer_library.fasta")
    rev_kmer_library_path = join(globals.temp_dir, "rev_kmer_library.fasta")
    merged_kmer_library_path = join(globals.temp_dir, "merged_kmer_library.txt")
    
    awk_script = r"""
        BEGIN {
            srand(seed);  # 设置随机数种子，默认为42
            skip_prob = 1 - p;  # 计算跳过概率
        }
        {
            # 处理奇数行（第一行编号为1）
            if (NR % 2 == 1) {
                current_pair = $0;  # 保存奇数行
                next;              # 跳过处理
            } else {
                # 偶数行：与上一行组成完整数据单位
                current_pair = current_pair ORS $0;
                
                # 随机决定是否保留这对数据
                if (rand() > skip_prob) {
                    print current_pair;
                }
            }
        }
    """
    command = f"jellyfish dump -L {min_multiplicity} {jf_path} | awk -v p={sample_fraction} -v seed={globals.seed} '{awk_script}' > {fwd_kmer_library_path}"
    logger.debug(f"Running Jellyfish dump command: {command}")
    subprocess.run(command, shell=True, check=True)

    command = f"seqkit seq -r -p -t DNA -j {globals.threads} {fwd_kmer_library_path} > {rev_kmer_library_path}"
    logger.debug(f"Creating reverse complement for kmer library: {command}")
    subprocess.run(command, shell=True, check=True)

    command = f"cat {fwd_kmer_library_path} {rev_kmer_library_path} | grep -v '^>' > {merged_kmer_library_path}"
    logger.debug(f"Running cat command: {command}")
    subprocess.run(command, shell=True, check=True)

    rev_input_path = join(globals.temp_dir, "rev_input.fasta")
    command = f"seqkit seq -r -p -t DNA -j {globals.threads} {fasta_path} > {rev_input_path}"
    logger.debug(f"Creating reverse complement for input FASTA: {command}")
    subprocess.run(command, shell=True, check=True)

    fwd_output_path = join(globals.temp_dir, "fwd_features.txt")
    rev_output_path = join(globals.temp_dir, "rev_features.txt")

    command = f"kmer_searcher {merged_kmer_library_path} {fasta_path} {fwd_output_path} {k} {globals.threads}"
    logger.debug(f"Searching kmers for forward strands: {command}")
    subprocess.run(command, shell=True, check=True)

    command = f"kmer_searcher {merged_kmer_library_path} {rev_input_path} {rev_output_path} {k} {globals.threads}"
    logger.debug(f"Searching kmers for reverse strands: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug("Parsing kmer_searcher output")
    for name, indices, counts in _parse_kmer_searcher_output(fwd_output_path):
        yield name, indices, counts, 0
    for name, indices, counts in _parse_kmer_searcher_output(rev_output_path):
        yield name, indices, counts, 1

# 使用示例
if __name__ == "__main__":
    # 异步启动任务
    input_fastq = "example.fastq"
    result = run_jellyfish(input_fastq, k=21, min_multiplicity=3)

    print("Jellyfish任务已启动，继续执行其他代码...")
    # 这里可以执行其他代码

    # 当需要结果时
    if result.done():
        print("任务已完成，结果如下:")
    else:
        print("等待任务完成...")

    # 获取结果（如果未完成会阻塞）
    for kmer, count in result.get_result():
        print(f"{kmer}: {count}")

    # 清理临时文件
    result.cleanup()

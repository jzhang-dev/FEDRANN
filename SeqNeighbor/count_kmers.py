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
    current_name = ""
    current_indices = []
    current_counts = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if current_name:
                    yield current_name, current_indices, current_counts
                current_name = line[1:].split(maxsplit=1)[0]
                current_indices = []
                current_counts = []
            else:
                idx, cnt = map(int, line.split())
                current_indices.append(idx)
                current_counts.append(cnt)
        if current_name:
            yield current_name, current_indices, current_counts


def count_lines(filename):
    """使用 wc -l 命令统计行数"""
    result = subprocess.run(
        ["wc", "-l", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wc failed: {result.stderr}")
    return int(result.stdout.split()[0])


def get_kmer_features(
    fasta_path: str, k: int, sample_fraction: float, min_multiplicity: int = 2
):
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
    logger.debug(
        f"Running Jellyfish count command: {' '.join(jellyfish_count_command)}"
    )
    subprocess.run(jellyfish_count_command, check=True)
    if not isfile(jf_path):
        raise RuntimeError(f"Jellyfish count output file not found: {jf_path}")

    fwd_kmer_library_path = join(globals.temp_dir, "fwd_kmer_library.fasta")
    rev_kmer_library_path = join(globals.temp_dir, "rev_kmer_library.fasta")

    awk_script = r"""
        BEGIN {
            srand(seed);  
            skip_prob = 1 - p;  
        }
        {
            if (NR % 2 == 1) {
                current_pair = $0;  
                next;              
            } else {
                current_pair = current_pair ORS $0;
                if (rand() > skip_prob) {
                    print current_pair;
                }
            }
        }
    """
    command = f"jellyfish dump -L {min_multiplicity} {jf_path} | awk -v p={sample_fraction} -v seed={globals.seed} '{awk_script}' > {fwd_kmer_library_path}"
    logger.debug(f"Running Jellyfish dump command: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug(f"Counting kmers in the library: {fwd_kmer_library_path}")
    kmer_count = count_lines(fwd_kmer_library_path) // 2  # 每个kmer有两行（header和sequence）
    logger.debug(f"Number of kmers in the library: {kmer_count}")

    command = f"seqkit seq -r -p -t DNA -j {globals.threads} {fwd_kmer_library_path} > {rev_kmer_library_path}"
    logger.debug(f"Creating reverse complement for kmer library: {command}")
    subprocess.run(command, shell=True, check=True)

    
    output_path = join(globals.temp_dir, "features.txt")

    command = f"cat {fwd_kmer_library_path} {rev_kmer_library_path} | grep -v '^>' | kmer_searcher /dev/stdin {fasta_path} {output_path} {k} {globals.threads}"
    logger.debug(f"Searching kmers for forward strands: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug("Parsing kmer_searcher output")
    for name, indices, counts in _parse_kmer_searcher_output(output_path):
        yield name, indices, counts, 0
        rev_indices = [i + kmer_count if i < kmer_count else i - kmer_count for i in indices]
        yield name, rev_indices, counts, 1


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

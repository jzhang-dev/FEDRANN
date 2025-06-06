import subprocess
import tempfile
import os
from os.path import abspath, isfile, join
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, Optional, Iterable
import struct
from collections import namedtuple
from typing import Iterator

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


def _parse_kmer_searcher_output(filename: str):
    with open(filename, "rb", buffering=1024 ** 3) as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError("文件头不完整")

        magic, version, reserved, total_records = struct.unpack("<4sB3sQ", header)
        
        if magic != b"KMER":
            raise ValueError("无效的文件格式")
        if version != 1:
            raise ValueError(f"不支持的版本号: {version}")

        for _ in range(total_records):
            id_len = struct.unpack("<H", f.read(2))[0]
            id_bytes = f.read(id_len)
            try:
                id_str = id_bytes.decode("utf-8")
            except UnicodeDecodeError:
                id_str = "".join(chr(b) if b < 128 else "_" for b in id_bytes)

            index_count = struct.unpack("<I", f.read(4))[0]
            data = f.read(8 * index_count)
            
            indices = struct.unpack(f"<{index_count}Q", data)
            # logger.debug(
            #     f"Parsed record: id={id_str}, indices={indices}"
            # )
            yield id_str, indices


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
    kmer_count = (
        count_lines(fwd_kmer_library_path) // 2
    )  # 每个kmer有两行（header和sequence）
    logger.debug(f"Number of kmers in the library: {kmer_count}")

    command = f"seqkit seq -r -p -t DNA -j {globals.threads} {fwd_kmer_library_path} > {rev_kmer_library_path}"
    logger.debug(f"Creating reverse complement for kmer library: {command}")
    subprocess.run(command, shell=True, check=True)

    kmer_searcher_output_dir = join(globals.temp_dir, "kmer_searcher")

    command = f"cat {fwd_kmer_library_path} {rev_kmer_library_path} | grep -v '^>' | kmer_searcher /dev/stdin {fasta_path} {kmer_searcher_output_dir} {k} {globals.threads}"
    logger.debug(f"Searching kmers for forward strands: {command}")
    subprocess.run(command, shell=True, check=True)

    logger.debug("Parsing kmer_searcher output")
    kmer_searcher_output_path = join(kmer_searcher_output_dir, "output.bin")
    if not isfile(kmer_searcher_output_path):
        raise RuntimeError(
            f"kmer_searcher output file not found: {kmer_searcher_output_path}"
        )
    for name, indices in _parse_kmer_searcher_output(kmer_searcher_output_path):
        yield name, indices, 0
        rev_indices = [
            i + kmer_count if i < kmer_count else i - kmer_count for i in indices
        ]
        yield name, rev_indices, 1

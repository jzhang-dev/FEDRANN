import subprocess
import tempfile
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, Optional

from .custom_logging import logger


class AsyncJellyfishResult:
    def __init__(self, temp_dir: str, future):
        self._temp_dir = temp_dir
        self._future = future
        self._result_file = os.path.join(temp_dir, "result.txt")

    def done(self) -> bool:
        """检查任务是否完成"""
        return self._future.done()

    def get_result(self) -> list[str]:
        """
        获取结果，如果任务未完成会阻塞直到完成
        返回生成器，产生(kmer, count)元组
        """

        kmers = self._future.result()  # 等待任务完成
        return kmers

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self._temp_dir):
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

        logger.debug(f"Running Jellyfish dump command: {' '.join(dump_cmd)}")
        p = subprocess.run(
            dump_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"Jellyfish dump command failed: {p.stderr.strip()}"
            )
        
        kmers = []
        for line in p.stdout.splitlines():
            kmer, count = line.strip().split(" ")
            kmers.append(kmer)
        return kmers
    
    # 使用线程池异步执行
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_jellyfish)

    # 返回结果对象
    return AsyncJellyfishResult(temp_dir, future)


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

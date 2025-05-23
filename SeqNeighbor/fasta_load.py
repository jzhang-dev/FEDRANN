import os, subprocess, logging
from os.path import isfile, join
from glob import glob
import json, pickle
from typing import (
    Mapping,
    Sequence,
    Optional,
    Collection,
    MutableMapping,
    Optional,
    Iterable,
    Iterator,
    overload,
    Literal,
    BinaryIO,
    TextIO,
    IO,
    Any,
    Generator,
    TypeVar,
    Generic,
    cast,
)
import heapq
from dataclasses import dataclass, field
import pysam
from pysam import AlignedSegment, PileupColumn
import concurrent.futures
from isal import igzip
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# from .custom_logging import logger

T = TypeVar("T")


@overload
def open_gzipped(
    path: str, mode: Literal["rt", "wt", "at"], gzipped: Optional[bool] = None, **kw
) -> TextIO:
    pass


@overload
def open_gzipped(
    path: str, mode: Literal["rb", "wb", "ab"], gzipped: Optional[bool] = None, **kw
) -> BinaryIO:
    pass


def open_gzipped(path, mode="rt", gzipped: Optional[bool] = None, **kw):
    if gzipped is None:
        gzipped = path.endswith(".gz")
    if gzipped:
        open_ = igzip.open
        return open_(path, mode)
    else:
        open_ = open
    return open_(path, mode, **kw)


def get_fastx_extension(file_path: str) -> str:
    extension_map: MutableMapping[str, str] = {
        "fasta": "fasta",
        "fa": "fasta",
        "fsa": "fasta",
        "fastq": "fastq",
        "fq": "fastq",
    }
    for extension, category in extension_map.items():
        if file_path.endswith(extension):
            return category
        if file_path.endswith(extension + ".gz"):
            return category + ".gz"
    else:
        raise ValueError(f"Invalid FASTX path: {file_path!r}")


def concatenate_fastx_files(input_paths: Sequence[str], output_path: str) -> None:
    if len(input_paths) < 2:
        raise ValueError("Requires two or more files to concatenate.")

    command = ["cat"] + list(input_paths) + [">", output_path]
    command_str = " ".join(command)
    subprocess.run(command_str, shell=True, check=True)


class FastxRecord:
    name: str
    sequence: str


@dataclass
class FastaRecord(FastxRecord):
    name: str
    sequence: str

    def __post_init__(self):
        self.sequence = self.sequence.upper()

    @staticmethod
    def from_lines(lines: Sequence[str]) -> 'FastaRecord':
        name = lines[0][1:-1].split()[0]
        sequence = ''.join(line.strip() for line in lines[1:])  # 合并多行序列
        return FastaRecord(name, sequence)


@dataclass
class FastqRecord(FastxRecord):
    name: str
    sequence: str
    quality: NDArray[np.uint8]

    @classmethod
    def from_lines(cls, lines: Sequence[str], offset: int = 33) -> "FastqRecord":
        name = lines[0][1:-1].split(" ")[0]
        sequence = lines[1][:-1]
        quality = np.array([ord(q) - offset for q in lines[3][:-1]], dtype=np.uint8)
        return cls(name, sequence, quality)


@dataclass
class _DataLoader(Generic[T]):
    file_path: str

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError
    
    def open(self):
        return self





class FastqLoader(_DataLoader[FastqRecord]):
    @staticmethod
    def _read_item(file_obj: TextIO) -> Iterator[Sequence[str]]:
        item = []
        for i, line in enumerate(file_obj):
            if line == "\n":
                # Skip empty lines
                i -= 1
                continue
            if i % 4 == 0 and i > 0:
                yield item
                item = [line]
            else:
                item.append(line)
        if len(item) == 4:
            yield item

    @staticmethod
    def _parse_item(item: Sequence[str]) -> FastqRecord:
        return FastqRecord.from_lines(item)

    def __iter__(self) -> Iterator[FastqRecord]:
        with open_gzipped(self.file_path, 'rt') as f:
            for _, item in enumerate(self._read_item(f)):
                yield self._parse_item(item)





@dataclass
class FastqReservoirSampler(FastqLoader):
    file_path: str
    sample_size: int
    temp_path: str
    memory_buffer_size: int = 1000
    seed: int = 109

    def __iter__(self) -> Iterator[FastqRecord]:
        rng = np.random.default_rng(self.seed)
        reservoir_heap: list[tuple[float, int]] = []  # 初始化用于存储蓄水池的堆队列
        memory_heap: list[tuple[float, int, Any]] = []  # 初始化用于存储内存缓存的堆队列
        offset_map: MutableMapping[int, int] = (
            {}
        )  # 偏移映射表，用于记录元素在硬盘缓存文件中的位置
        sample_size = self.sample_size

        with open(self.temp_path, "wt") as tf, open_gzipped(self.file_path, 'rt') as f:
            for i, item in enumerate(self._read_item(f)):
                y = rng.random()  # 生成随机特征值
                accept = False
                if len(reservoir_heap) < sample_size:
                    accept = True  # 如果采样堆未满，接受当前元素
                    heapq.heappush(reservoir_heap, (y, i))
                elif y > reservoir_heap[0][0]:
                    # 如果随机特征值大于堆顶元素，接受当前元素，并替换堆顶（特征值最小）元素
                    accept = True
                    y_popped, _ = heapq.heappushpop(reservoir_heap, (y, i))
                    if y_popped >= memory_heap[0][0]:
                        # 如果被替换的元素存在于内存缓存中，则从内存缓存中删除该元素
                        heapq.heappop(memory_heap)

                if accept:
                    heapq.heappush(memory_heap, (y, i, item))  # 将当前元素加入内存缓存
                    if len(memory_heap) >= self.memory_buffer_size:
                        # 获取数据堆中最大的元素并写入硬盘缓存
                        y, j, item = heapq.nlargest(1, memory_heap, key=lambda x: x[0])[
                            0
                        ]
                        memory_heap.remove((y, j, item))
                        offset = tf.tell()
                        tf.write("".join(item))
                        offset_map[j] = offset  # 记录元素在硬盘缓存文件中的位置

            memory_buffer = {i: item for _, i, item in memory_heap}

        with open(self.temp_path, "rt") as tf:
            for _, i in reservoir_heap:
                if i in memory_buffer:
                    item = memory_buffer[i]
                    record = self._parse_item(item)
                else:
                    tf.seek(offset_map[i])  # 从硬盘缓存文件中读取元素
                    item = next(iter(self._read_item(tf)))
                    record = self._parse_item(item)
                yield record



class FastaLoader(_DataLoader[FastaRecord]):
    @staticmethod
    def _read_item(file_obj: TextIO) -> Iterator[Sequence[str]]:
        item = []
        for line in file_obj:
            if line.startswith('>'):
                if item:  # 如果已经有数据，先返回
                    yield item
                    item = []
                item.append(line)  # 添加 header
            else:
                item.append(line)  # 添加序列行
        if item:  # 返回最后一个记录
            yield item

    @staticmethod
    def _parse_item(item: Sequence[str]) -> FastaRecord:
        return FastaRecord.from_lines(item)

    def __iter__(self) -> Iterator[FastaRecord]:
        with open_gzipped(self.file_path, 'rt') as f:
            for item in self._read_item(f):
                yield self._parse_item(item)


# def make_fasta_index(file_path: str, index_path: Optional[str] = None) -> None:
#     """
#     创建 FASTA 文件的索引。
    
#     参数：
#         file_path: FASTA 文件路径
#         index_path: 索引文件路径，默认为 file_path + ".fai"
    
#     异常：
#         如果索引文件已存在，则抛出 FileExistsError
#     """
#     # TODO: 处理压缩文件
#     if index_path is None:
#         index_path = file_path + ".fai"
#     else:
#         parent_dir = os.path.dirname(index_path)
#         symlink_path = os.path.join(parent_dir, "input.fasta")
#         os.system(f"ln -s {file_path} {symlink_path}")
#         file_path = symlink_path
#     if isfile(index_path):
#         raise FileExistsError(f"Index file already exists: {index_path!r}")
#     logger.debug(f"Creating index for {file_path!r} at {index_path!r}")
#     pysam.faidx(file_path)
#     if index_path != file_path + ".fai":
#         os.rename(file_path + ".fai", index_path)
#     if not isfile(index_path):
#         raise RuntimeError(f"Failed to create index file: {index_path!r}")


# class IndexedFastaLoader(FastaLoader):
#     def __init__(self, file_path: str, index_path: Optional[str] = None):
#         self.file_path = file_path
#         self.index_path = index_path or file_path + ".fai"
#         if not isfile(self.index_path):
#             raise FileNotFoundError(
#                 f"Index file not found: {self.index_path!r}"
#             )
        
#     def get_sequence(self, name: str, start: int | None = None, end: int | None = None) -> FastaRecord:
#         fasta = pysam.FastaFile(self.file_path, filepath_index=self.index_path)
#         seq = fasta.fetch(reference=name, start=start, end=end)
#         record = FastaRecord(name=name, sequence=seq)
#         return record


# @dataclass
# class ReferenceSequences:
#     fasta_path: str
#     fai_path: str
#     _sequences: MutableMapping[str, FastaRecord] = field(default_factory=dict, init=False)

#     def __post_init__(self):
#         self._loader = IndexedFastaLoader(self.fasta_path, self.fai_path)

#     def load_sequence(self, name: str) -> FastaRecord:
#         logger.debug(f"Loading sequence {name}")
#         record = self._loader.get_sequence(name)
#         self._sequences[name] = record
#         return record
    
#     def prune_sequence(self, name: str) -> None:
#         logger.debug(f"Pruning sequence {name}")
#         if name in self._sequences:
#             del self._sequences[name]

#     def __getitem__(self, name: str) -> FastaRecord:
#         if name not in self._sequences:
#             self.load_sequence(name)
#         return self._sequences[name]




@dataclass
class FastaReservoirSampler(FastaLoader):
    file_path: str
    sample_size: int
    temp_path: str
    memory_buffer_size: int = 1000
    seed: int = 455

    def __iter__(self) -> Iterator[FastaRecord]:
        rng = np.random.default_rng(self.seed)
        reservoir_heap: list[tuple[float, int]] = []  # 初始化用于存储蓄水池的堆队列
        memory_heap: list[tuple[float, int, Any]] = []  # 初始化用于存储内存缓存的堆队列
        offset_map: MutableMapping[int, int] = (
            {}
        )  # 偏移映射表，用于记录元素在硬盘缓存文件中的位置
        sample_size = self.sample_size

        with open(self.temp_path, "wt") as tf, open_gzipped(self.file_path, 'rt') as f:
            for i, item in enumerate(self._read_item(f)):
                y = rng.random()  # 生成随机特征值
                accept = False
                if len(reservoir_heap) < sample_size:
                    accept = True  # 如果采样堆未满，接受当前元素
                    heapq.heappush(reservoir_heap, (y, i))
                elif y > reservoir_heap[0][0]:
                    # 如果随机特征值大于堆顶元素，接受当前元素，并替换堆顶（特征值最小）元素
                    accept = True
                    y_popped, _ = heapq.heappushpop(reservoir_heap, (y, i))
                    if y_popped >= memory_heap[0][0]:
                        # 如果被替换的元素存在于内存缓存中，则从内存缓存中删除该元素
                        heapq.heappop(memory_heap)

                if accept:
                    heapq.heappush(memory_heap, (y, i, item))  # 将当前元素加入内存缓存
                    if len(memory_heap) >= self.memory_buffer_size:
                        # 获取数据堆中最大的元素并写入硬盘缓存
                        y, j, item = heapq.nlargest(1, memory_heap, key=lambda x: x[0])[
                            0
                        ]
                        memory_heap.remove((y, j, item))
                        offset = tf.tell()
                        tf.write("".join(item))
                        offset_map[j] = offset  # 记录元素在硬盘缓存文件中的位置

            memory_buffer = {i: item for _, i, item in memory_heap}

        with open(self.temp_path, "rt") as tf:
            for _, i in reservoir_heap:
                if i in memory_buffer:
                    item = memory_buffer[i]
                    record = self._parse_item(item)
                else:
                    tf.seek(offset_map[i])  # 从硬盘缓存文件中读取元素
                    item = next(iter(self._read_item(tf)))
                    record = self._parse_item(item)
                yield record



class BamSegmentLoader(_DataLoader[AlignedSegment]):    
    def __iter__(self) -> Iterator[AlignedSegment]:
        for segment in pysam.AlignmentFile(self.file_path, 'rb'):
            yield segment

@dataclass
class BamSegmentReservoirSampler(_DataLoader[AlignedSegment]):
    file_path: str
    sample_size: int
    reservoir_path: str
    memory_buffer_size: int = 1000
    seed: int = 1053

    def __iter__(self) -> Iterator[AlignedSegment]:
        rng = np.random.default_rng(self.seed)
        reservoir_heap: list[tuple[float, int]] = []  # 初始化用于存储蓄水池的堆队列
        memory_heap: list[tuple[float, int, Any]] = []  # 初始化用于存储内存缓存的堆队列
        offset_map: MutableMapping[int, int] = (
            {}
        )  # 偏移映射表，用于记录元素在硬盘缓存文件中的位置
        sample_size = self.sample_size

        with pysam.AlignmentFile(self.file_path, 'rb') as f:
            with pysam.AlignmentFile(self.reservoir_path, "wb", header=f.header) as tf:
                for i, segment in enumerate(f):
                    if segment.is_secondary:
                        continue
                    y = rng.random()  # 生成随机特征值
                    accept = False
                    if len(reservoir_heap) < sample_size:
                        accept = True  # 如果采样堆未满，接受当前元素
                        heapq.heappush(reservoir_heap, (y, i))
                    elif y > reservoir_heap[0][0]:
                        # 如果随机特征值大于堆顶元素，接受当前元素，并替换堆顶（特征值最小）元素
                        accept = True
                        y_popped, _ = heapq.heappushpop(reservoir_heap, (y, i))
                        if y_popped >= memory_heap[0][0]:
                            # 如果被替换的元素存在于内存缓存中，则从内存缓存中删除该元素
                            heapq.heappop(memory_heap)

                    if accept:
                        heapq.heappush(memory_heap, (y, i, segment))  # 将当前元素加入内存缓存
                        if len(memory_heap) >= self.memory_buffer_size:
                            # 获取数据堆中最大的元素并写入硬盘缓存
                            y, j, segment = heapq.nlargest(1, memory_heap, key=lambda x: x[0])[
                                0
                            ]
                            memory_heap.remove((y, j, segment))
                            offset = tf.tell()
                            tf.write(segment)
                            offset_map[j] = offset  # 记录元素在硬盘缓存文件中的位置

            memory_buffer = {i: item for _, i, item in memory_heap}

        reservoir_heap = sorted(reservoir_heap, key=lambda x: x[1]) # 将 reservoir_heap 按照索引排序
        with pysam.AlignmentFile(self.reservoir_path, "rb") as tf:
            for _, i in reservoir_heap:
                if i in memory_buffer:
                    segment = memory_buffer[i]
                else:
                    tf.seek(offset_map[i])  # 从硬盘缓存文件中读取元素
                    segment = next(iter(tf))
                yield segment


# def save_json(obj: Any, filename: str) -> None:
#     """
#     将 Python 对象保存为 JSON 文件。
    
#     参数：
#     - obj: 要保存的 Python 对象
#     - filename: 保存的文件名，如 "data/myobject.json"
    
#     异常：
#     - 如果文件无法写入，会抛出 IOError
#     """
#     with open(filename, 'wt', encoding='utf-8') as f:
#         json.dump(obj, f, ensure_ascii=False, indent=4)
def save_json(obj: Any, filename: str) -> None:
    """
    将 Python 对象保存为 JSON 文件。
    
    参数：
    - obj: 要保存的 Python 对象
    - filename: 保存的文件名，如 "data/myobject.json"
    
    异常：
    - 如果文件无法写入，会抛出 IOError
    """
    with open(filename, 'wt', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=None, separators=(',', ':'))



def save_dataframe_to_hdf(df: pd.DataFrame, path: str, key: str = 'data'):
    """
    保存 DataFrame 和其 attrs 到 HDF5 文件中。
    
    参数：
        df  : 要保存的 DataFrame，支持 df.attrs
        path: HDF5 文件路径（如 'data.h5'）
        key : 存储在 HDF5 中的键名（如 'mydata'）
    """
    with pd.HDFStore(path) as store:
        store.put(key, df)
        storer = store.get_storer(key) # type: ignore
        for k, v in df.attrs.items():
            storer.attrs[k] = v


def load_dataframe_from_hdf(path: str, key: str = 'data') -> pd.DataFrame:
    """
    从 HDF5 文件中加载 DataFrame 和其 attrs。
    
    参数：
        path: HDF5 文件路径（如 'data.h5'）
        key : 存储在 HDF5 中的键名（如 'mydata'）
    
    返回：
        带 attrs 的 DataFrame
    """
    with pd.HDFStore(path) as store:
        df = store[key]
        storer = store.get_storer(key) # type: ignore
        df.attrs = dict(storer.attrs)
    return df # type: ignore
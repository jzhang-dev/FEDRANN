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
from dataclasses import dataclass, field
from isal import igzip
import numpy as np
from numpy.typing import NDArray

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
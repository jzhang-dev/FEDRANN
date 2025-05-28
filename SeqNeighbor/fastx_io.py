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
    NamedTuple,
    cast,
)
from dataclasses import dataclass, field
from isal import igzip
import numpy as np
from numpy.typing import NDArray


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


def init_reverse_complement():
    TRANSLATION_TABLE = str.maketrans("ACTGactg", "TGACtgac")

    def reverse_complement(sequence: str) -> str:
        """
        >>> reverse_complement("AATC")
        'GATT'
        >>> reverse_complement("CCANT")
        'ANTGG'
        """
        sequence = str(sequence)
        return sequence.translate(TRANSLATION_TABLE)[::-1]

    return reverse_complement


reverse_complement = init_reverse_complement()


class FastxRecord(NamedTuple):
    name: str
    sequence: str
    orientation: int = 0

def get_reverse_complement_record(record: FastxRecord) -> FastxRecord:
    """
    Get the reverse complement of a FastxRecord.
    """
    return FastxRecord(
        name=record.name,
        sequence=reverse_complement(record.sequence),
        orientation=1 - record.orientation,
    )


@dataclass
class _DataLoader(Generic[T]):
    file_path: str

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def open(self):
        return self


class FastqLoader(_DataLoader[FastxRecord]):
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
    def _parse_item(item: Sequence[str]) -> FastxRecord:
        lines = item
        name = lines[0][1:-1].split(" ")[0]
        sequence = lines[1][:-1]
        return FastxRecord(name=name, sequence=sequence)

    def __iter__(self) -> Iterator[FastxRecord]:
        with open_gzipped(self.file_path, "rt") as f:
            for _, item in enumerate(self._read_item(f)):
                yield self._parse_item(item)


class FastaLoader(_DataLoader[FastxRecord]):
    @staticmethod
    def _read_item(file_obj: TextIO) -> Iterator[Sequence[str]]:
        item = []
        for line in file_obj:
            if line.startswith(">"):
                if item:  # 如果已经有数据，先返回
                    yield item
                    item = []
                item.append(line)  # 添加 header
            else:
                item.append(line)  # 添加序列行
        if item:  # 返回最后一个记录
            yield item

    @staticmethod
    def _parse_item(item: Sequence[str]) -> FastxRecord:
        lines = item
        name = lines[0][1:-1].split()[0]
        sequence = "".join(line.strip() for line in lines[1:])  # 合并多行序列
        return FastxRecord(name=name, sequence=sequence)

    def __iter__(self) -> Iterator[FastxRecord]:
        with open_gzipped(self.file_path, "rt") as f:
            for item in self._read_item(f):
                yield self._parse_item(item)

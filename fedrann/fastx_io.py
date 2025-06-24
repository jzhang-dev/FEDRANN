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
import pysam

from .custom_logging import logger


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



def convert_fastq_to_fasta(fastq_path: str, fasta_path: str, threads: int) -> None:
    """
    Convert a FASTQ file to a FASTA file.
    """
    seqkit_command = [
        "seqkit", "fq2fa",
        "-j", f"{threads}",
        fastq_path,
        "-o", fasta_path
    ]
    logger.debug(f"Running seqkit command: {' '.join(seqkit_command)}")
    subprocess.run(seqkit_command, check=True)
    if not isfile(fasta_path):
        raise RuntimeError(f"Failed to convert FASTQ to FASTA: {fasta_path} not found after conversion.")
    

def unzip(input_path: str, output_path: str) -> None:
    """
    Unzip a gzipped file.
    """
    if not input_path.endswith(".gz"):
        raise ValueError(f"Input path {input_path} is not a gzipped file.")
    
    logger.debug(f"Unzipping {input_path} to {output_path}")
    gunzip_command = ["gunzip", "-c", input_path]
    with open(output_path, "wb") as out_f:
        subprocess.run(gunzip_command, stdout=out_f, check=True)
    if not isfile(output_path):
        raise RuntimeError(f"Failed to unzip {input_path}: {output_path} not found after unzipping.")
    

def make_fasta_index(file_path: str, index_path: Optional[str] = None) -> None:
    """
    创建 FASTA 文件的索引。
    
    参数：
        file_path: FASTA 文件路径
        index_path: 索引文件路径，默认为 file_path + ".fai"
    
    异常：
        如果索引文件已存在，则抛出 FileExistsError
    """
    # TODO: 处理压缩文件
    if index_path is None:
        index_path = file_path + ".fai"
    else:
        parent_dir = os.path.dirname(index_path)
        symlink_path = os.path.join(parent_dir, "input.fasta")
        os.system(f"ln -s {file_path} {symlink_path}")
        file_path = symlink_path
    if isfile(index_path):
        raise FileExistsError(f"Index file already exists: {index_path!r}")
    logger.debug(f"Creating index for {file_path!r} at {index_path!r}")
    pysam.faidx(file_path)
    if index_path != file_path + ".fai":
        os.rename(file_path + ".fai", index_path)
    if not isfile(index_path):
        raise RuntimeError(f"Failed to create index file: {index_path!r}")
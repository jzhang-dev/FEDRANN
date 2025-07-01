from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy as np

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

include_dirs = ["../external/robin-hood-hashing/src/include", current_dir, np.get_include()]

extensions = [
    Extension(
        "Kmer_searcher",
        sources=["Ckmer_searcher.pyx", "Kmer_searcher/Kmer_searcher.cpp"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
        extra_link_args=["-std=c++11"],
        include_dirs=include_dirs
    )
]

setup(
    name="Kmer_searcher",
    version="0.0.1",
    author="Junyi He",
    author_email="hejunyi@genomics.cn",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
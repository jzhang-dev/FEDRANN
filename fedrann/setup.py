from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

include_dirs = ["../external/robin-hood-hashing/src/include", current_dir]

extensions = [
    Extension(
        "Kmer_searcher",
        sources=["Ckmer_searcher.pyx", "Kmer_searcher/Kmer_searcher.cpp"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-march=native"],
        extra_link_args=[],
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
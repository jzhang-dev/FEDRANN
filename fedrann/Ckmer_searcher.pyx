# cython: language_level=3
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport numpy as np
import numpy as np
np.import_array()  # 初始化NumPy C-API

# 使用 memoryview 接口
from cpython cimport PyObject, Py_INCREF
from cython.view cimport array as cvarray

cdef extern from "Kmer_searcher/Kmer_searcher.h":
    cdef cppclass KmerSearcher:
        KmerSearcher(int k) except +
        void load_kmer_libs(const string&, const string&)
        void process_sequences(const string&, const string&, int)
        const uint64_t* get_col_indices_ptr() const
        size_t get_col_indices_size() const
        const uint64_t* get_indptr_ptr() const
        size_t get_indptr_size() const

cdef class PyKmerSearcher:
    cdef KmerSearcher* c_searcher
    
    def __cinit__(self, int k):
        self.c_searcher = new KmerSearcher(k)
    
    def __dealloc__(self):
        del self.c_searcher
    
    def load_kmer_libs(self, str kmer_lib_path1, str kmer_lib_path2):
        self.c_searcher.load_kmer_libs(
            kmer_lib_path1.encode('utf-8'),
            kmer_lib_path2.encode('utf-8')
        )
    
    def process_sequences(self, str input_file, str output_seqid_file, int num_threads):
        self.c_searcher.process_sequences(
            input_file.encode('utf-8'),
            output_seqid_file.encode('utf-8'),
            num_threads
        )
    
    def get_results(self):
        # 获取指针和大小
        cdef const uint64_t* col_ptr = self.c_searcher.get_col_indices_ptr()
        cdef size_t col_size = self.c_searcher.get_col_indices_size()
        cdef const uint64_t* indptr_ptr = self.c_searcher.get_indptr_ptr()
        cdef size_t indptr_size = self.c_searcher.get_indptr_size()
        
        # 使用 memoryview 接口创建数组
        cdef np.ndarray col_arr = self.create_array_from_ptr(col_ptr, col_size)
        cdef np.ndarray indptr_arr = self.create_array_from_ptr(indptr_ptr, indptr_size)
        
        return col_arr, indptr_arr
    
    cdef np.ndarray create_array_from_ptr(self, const uint64_t* data, size_t size):
        """通过内存视图创建NumPy数组"""
        if size == 0:
            return np.array([], dtype=np.uint64)
        
        # 创建Cython内存视图
        cdef uint64_t[::1] mv = <uint64_t[:size]> data
        
        # 转换为NumPy数组（零复制）
        cdef np.ndarray arr = np.asarray(mv)
        
        # 增加对象的引用计数
        Py_INCREF(self)
        
        return arr
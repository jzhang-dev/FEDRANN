# cython: language_level=3
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.stdint cimport uint64_t


cdef extern from "Kmer_searcher/Kmer_searcher.h":
    cdef cppclass KmerSearcher:
        KmerSearcher(int k) except +
        void load_kmer_libs(const string&, const string&)
        void process_sequences(const string&, int)
        vector[pair[string, vector[uint64_t]]] get_results()

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
    
    def process_sequences(self, str input_file, int num_threads):
        self.c_searcher.process_sequences(
            input_file.encode('utf-8'),
            num_threads
        )
    
    def get_results(self):
        cdef vector[pair[string, vector[uint64_t]]] results = self.c_searcher.get_results()
        py_results = []
        
        for result in results:
            id_str = result.first.decode('utf-8')
            indices = [index for index in result.second]
            py_results.append((id_str, indices))
        
        return py_results
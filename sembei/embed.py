import math
import multiprocessing

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from .embed_inner import construct_cooccurrence_matrix, construct_cooccurrence_matrix_widecontext


class OSCCASembei(object):
    '''Implementation of segmentation-free version of OSCCA [押切+2017]_

    Parameters
    --------------------
    string : str
        a raw text corpus
    vocabulary : list of str
        list of character n-gram
    dim : int
        a dimension of vector representation
    verbose : bool (default=False)
    n_iter_rsvd
    wide_window
    size_window
    inc

    Notes
    --------------------
    .. [押切+2017]
        押切 孝将, 下平 英寿. (2017).
        **単語分割を経由しない単語埋め込み**.
        言語処理学会第23回年次大会論文集, pp.258-261.
    '''

    def __init__(self, string, vocabulary, dim,
                 verbose=False, n_iter_rsvd=3,
                 wide_window=False, size_window=1, inc=1):
        self.string = string
        self.vocabulary = vocabulary
        self.dim = dim
        self.verbose = verbose
        self.n_iter_rsvd = n_iter_rsvd
        self.wide_window = wide_window
        self.size_window = size_window
        self.inc = inc

        self.size_vocabulary = len(vocabulary)
        self.dict_vocabulary = dict(zip(vocabulary, range(self.size_vocabulary)))
        self.max_n_ngram = max(len(s) for s in self.vocabulary)

        if self.verbose:
            messages = '''Parameters :
            size_vocabulary : {size_vocabulary}
            dim             : {dim}
            size_window     : {size_window}
            wide_window     : {wide_window}
            n_iter_rsvd     : {n_iter_rsvd}
            max_n_ngram     : {max_n_ngram}
            inc             : {inc}
            '''
            print(messages.format(**self.__dict__))

    def construct_cooccurrence_matrix(self, n_cores=1, n_chunk=1):
        '''共起行列を構成する

        Parameters
        ------------------------------
        n_cores : integer (default=1)
            Number of cores
        n_chunk : integer (default=1)
            Number of chunks

        Notes
        ------------------------------
        文字列 (self.string) から共起行列を構成する処理には
        starmap を使っているが，<http://bugs.python.org/issue17560>
        という問題がある．
        そこで，construct_cooccurrence_matrix{,_widecontext}
        の返り値(の行列の非ゼロ要素数)が小さくなるよう
        チャンクに切ってから starmap に投げる必要がある．

        チャンクに切って実行する (i.e. `n_chunk > 1`) 場合，
        チャンクの境界にある n-gram を何個か無視することになるが，
        これは気にしないことにする．
        '''
        if self.wide_window:
            construct_cooccurrence = construct_cooccurrence_matrix_widecontext
        else:
            construct_cooccurrence = construct_cooccurrence_matrix

        nrow = self.size_vocabulary
        ncol = self.size_vocabulary * (int(self.wide_window) + 2)
        self.G1_diag = np.zeros(nrow)
        self.count_matrix = sparse.csc_matrix((nrow, ncol), dtype=np.int64)

        size_string = len(self.string)
        size_chunk_string = math.ceil(size_string / n_chunk)
        n_chunk_pool = math.ceil(n_chunk / n_cores)
        size_chunk_pool = math.ceil(n_chunk / n_chunk_pool)
        args = []

        for i_chunk in range(n_chunk):
            i_start = i_chunk * size_chunk_string
            i_end = min((i_chunk + 1) * size_chunk_string - 1, size_string)
            args.append((self.string[i_start:i_end], self.dict_vocabulary,
                         self.size_window, self.size_vocabulary,
                         self.max_n_ngram, self.inc))

            if ((i_chunk + 1) % n_cores == 0) or (i_chunk == n_chunk - 1):
                try:
                    pool = multiprocessing.Pool(n_cores)
                    callback = pool.starmap(construct_cooccurrence, args)
                    args = []
                finally:
                    pool.close()
                    pool.join()

                for c in callback:
                    self.G1_diag += c[0]
                    self.count_matrix += c[1]

        self.G2_diag = self.count_matrix.sum(axis=0).A.ravel()

        # Print some informations
        if self.verbose:
            nrow, ncol = self.count_matrix.shape
            nnz = self.count_matrix.nnz
            density = 100 * nnz / (nrow * ncol)

            print('nnz     : {0}'.format(nnz))
            print('size    : {0}x{1}'.format(nrow, ncol))
            print('density : {0}'.format(density))
            print('\n')

    def compute(self):
        A = (sparse.diags(self.G1_diag**(-1 / 4))
             @ self.count_matrix.power(1 / 2)
             @ sparse.diags(self.G2_diag**(-1 / 4)))

        self.U, self.S_diag, self.VT = randomized_svd(
            A, n_components=self.dim, n_iter=self.n_iter_rsvd)

    def get_vectors(self, normalize_vectors=True, gamma=0):
        vectors = sparse.diags((self.G1_diag + gamma)**(-1 / 4)) @ self.U
        if normalize_vectors:
            vectors = normalize(vectors)

        return pd.DataFrame(vectors, index=self.vocabulary)

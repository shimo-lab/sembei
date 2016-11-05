import math
import multiprocessing
import datetime as dt

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from .embed_inner import construct_cooccurrence_matrix


class Sembei(object):
    '''Segmentation-free word embedding method

    Parameters
    --------------------
    string
    vocabulary
    dim
    context
        width of context window
    verbose_level
        0 : all (for a small corpus)
        1 : a little
        2 : never
    '''

    def __init__(self, string, vocabulary, dim, size_window=1,
                 verbose_level=1, n_iter_rsvd=3):
        self.string = string
        self.vocabulary = vocabulary
        self.dim = dim
        self.size_window = size_window
        self.verbose_level = verbose_level
        self.n_iter_rsvd = n_iter_rsvd

        self.size_vocabulary = len(vocabulary)
        self.dict_vocabulary = dict(zip(self.vocabulary, range(self.size_vocabulary)))
        self.max_n_ngram = max(len(s) for s in self.vocabulary)

        if self.verbose_level <= 1:
            messages = '''
            size_vocabulary : {size_vocabulary}
            dim : {dim}
            '''

            print(dt.datetime.today())
            print(messages.format(**self.__dict__))

    def each_construct_matrices(self, pid, substring, size_chunk,
                                n_cores, size_string):

        retval = construct_cooccurrence_matrix(
            substring, self.dict_vocabulary,
            self.size_window, self.size_vocabulary, self.max_n_ngram)

        return retval

    def construct_cooccurrence_matrix(self, n_cores=1):
        '''
        n_cores
        '''
        size_string = len(self.string)
        size_chunk = math.ceil(size_string / n_cores)

        args = []

        for pid in range(n_cores):
            i_start = pid * size_chunk

            if pid == n_cores - 1:
                i_end = size_string
            else:
                # チャンクの境界にある n-gram を何個か無視することになるが気にしない
                i_end = (pid + 1) * size_chunk - 1

            args.append((pid, self.string[i_start:i_end],
                         size_chunk, n_cores, size_string))

        try:
            pool = multiprocessing.Pool(n_cores)
            callback = pool.starmap(self.each_construct_matrices, args)
        finally:
            pool.close()
            pool.join()

        nrow_H = self.size_vocabulary
        ncol_H = 2 * self.size_window * self.size_vocabulary

        self.G1_diag = np.zeros(nrow_H)
        self.count_matrix = sparse.csc_matrix((nrow_H, ncol_H))

        for c in callback:
            self.G1_diag += c[0]
            self.count_matrix += c[1]

        self.G2_diag = self.count_matrix.sum(axis=0).A.ravel()

        # Print some informations
        if self.verbose_level <= 1:
            nrow, ncol = self.count_matrix.shape
            nnz = self.count_matrix.nnz
            density = 100 * nnz / (nrow * ncol)

            print('# of nonzero : {0}'.format(nnz))
            print('size : {0}x{1}'.format(nrow, ncol))
            print('density : {0}'.format(density))
            print('\n')

    def compute(self):
        A = (sparse.diags(self.G1_diag**(-1 / 4))
             @ self.count_matrix.power(1 / 2)
             @ sparse.diags(self.G2_diag**(-1 / 4)))

        self.U, self.S_diag, self.VT = randomized_svd(
            A, n_components=self.dim, n_iter=self.n_iter_rsvd)

        if self.verbose_level <= 1:
            plt.plot(self.S_diag, 'b+')
            plt.xlim(-2, )
            plt.yscale('log')
            plt.xlabel('$i$', fontsize=20)
            plt.ylabel('singular values $\sigma_i$', fontsize=20)
            plt.show()

    def get_vectors(self, normalize_vectors=True, gamma=0):

        vectors = sparse.diags((self.G1_diag + gamma)**(-1 / 4)) @ self.U

        if normalize_vectors:
            vectors = normalize(vectors)

        return pd.DataFrame(vectors, index=self.vocabulary)

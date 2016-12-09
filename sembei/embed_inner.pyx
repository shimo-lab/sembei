import numpy as np
cimport numpy as np
from scipy import sparse
from cython import boundscheck

cdef long SIZE_ARRAY = 10000000


def construct_cooccurrence_matrix(
        str lines_str, dict dict_vocabulary_all,
        long size_window, long size_vocabulary_all, long width_ngram_context):
    cdef:
        np.ndarray[np.int64_t, ndim = 1] indices_row = np.empty(SIZE_ARRAY, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] indices_col = np.empty(SIZE_ARRAY, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] ngram_counts = np.zeros(size_vocabulary_all, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] velues = np.ones(SIZE_ARRAY, dtype=int)
        long n_added = 0
        long n_characters = len(lines_str)
        long width, id_ngram, id_ngram_next, i, i_start
        long size_ngram, size_ngram_next

    count_matrix = sparse.csc_matrix((size_vocabulary_all, 2 * size_vocabulary_all))

    for i in range(n_characters):

        for size_ngram in range(1, width_ngram_context + 1):
            id_ngram = dict_vocabulary_all.get(lines_str[i:(i + size_ngram)], -1)

            if id_ngram < 0:
                continue

            ngram_counts[id_ngram] += 1
            i_start = i + size_ngram

            if i_start >= n_characters:
                continue

            for size_ngram_next in range(1, width_ngram_context + 1):
                id_ngram_next = dict_vocabulary_all.get(
                    lines_str[i_start:(i_start + size_ngram_next)], -1)

                if id_ngram_next < 0:
                    continue

                # Right context
                indices_row[n_added] = id_ngram
                indices_col[n_added] = id_ngram_next + size_vocabulary_all

                # Left context
                indices_row[n_added + 1] = id_ngram_next
                indices_col[n_added + 1] = id_ngram

                n_added += 2

        if (n_added + 2 * width_ngram_context**2 > SIZE_ARRAY) or (i >= n_characters - 1):
            index_added = np.arange(n_added, dtype=int)
            count_matrix_temp = sparse.csc_matrix(
                (velues[index_added], (indices_row[index_added], indices_col[index_added])),
                shape=(size_vocabulary_all, 2 * size_vocabulary_all))
            count_matrix += count_matrix_temp
            n_added = 0

    return (ngram_counts, count_matrix)


#@boundscheck(False)
def construct_cooccurrence_matrix_widecontext(
        str lines_str, dict dict_vocabulary_all,
        long size_window, long size_vocabulary_all, long width_ngram_context):
    cdef:
        np.ndarray[np.int64_t, ndim = 1] indices_row = np.empty(SIZE_ARRAY, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] indices_col = np.empty(SIZE_ARRAY, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] ngram_counts = np.zeros(size_vocabulary_all, dtype=int)
        np.ndarray[np.int64_t, ndim = 1] velues = np.ones(SIZE_ARRAY, dtype=int)
        long n_added = 0
        long n_characters = len(lines_str)
        long width, id_ngram, id_ngram_next, i, i_start, i_end, i_width
        long size_ngram, size_ngram_next
        long nrow_count_matrix = size_vocabulary_all
        long ncol_count_matrix = 4 * size_vocabulary_all

    count_matrix = sparse.csc_matrix((nrow_count_matrix, ncol_count_matrix))

    for i in range(n_characters):

        for size_ngram in range(1, width_ngram_context + 1):
            id_ngram = dict_vocabulary_all.get(lines_str[i:(i + size_ngram)], -1)

            if id_ngram < 0:
                continue

            ngram_counts[id_ngram] += 1
            i_start = i + size_ngram

            if i_start >= n_characters:
                continue

            for width in range(1, size_window + 1):

                i_end = i_start + width

                if i_end >= n_characters:
                    continue

                for i_width in range(1, width + 1):
                    id_ngram_next = dict_vocabulary_all.get(
                        lines_str[(i_end - i_width):i_end], -1)

                    if id_ngram_next < 0:
                        continue

                    # id_ngram と連続している文字 n-gram の場合
                    if i_width == width:
                        # Right context
                        indices_row[n_added] = id_ngram
                        indices_col[n_added] = id_ngram_next + 2 * size_vocabulary_all

                        # Left context
                        indices_row[n_added + 1] = id_ngram_next
                        indices_col[n_added + 1] = id_ngram + size_vocabulary_all

                    else:
                        # Right context
                        indices_row[n_added] = id_ngram
                        indices_col[n_added] = id_ngram_next + 3 * size_vocabulary_all

                        # Left context
                        indices_row[n_added + 1] = id_ngram_next
                        indices_col[n_added + 1] = id_ngram

                    n_added += 2

        if (n_added + 2 * width_ngram_context**3 > SIZE_ARRAY) or (i >= n_characters - 1):
            index_added = np.arange(n_added, dtype=int)
            count_matrix_temp = sparse.csc_matrix(
                (velues[index_added], (indices_row[index_added], indices_col[index_added])),
                shape=(nrow_count_matrix, ncol_count_matrix))
            count_matrix += count_matrix_temp
            n_added = 0

    return (ngram_counts, count_matrix)

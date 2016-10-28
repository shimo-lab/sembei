from collections import namedtuple
from cython import cdivision


@cdivision(False)
def lossycounting_ngram(str string, long n_ngram, double epsilon, double support_threshold):
    '''Implementation of lossy counting algorithm [Manku-Motwani2002]_ for character n-grams from a long string
    
    Parameters
    --------------------
    string : str
    n_ngram : long
    epsilon : double
    support_threshold : double
    
    
    Notes
    --------------------
    .. [Manku-Motwani2002]
        Gurmeet Singh Manku & Rajeev Motwani. (2002)
        **Approximate Frequency Counts over Data Streams**.
        VLDB'02, 346 - 357.
        http://dl.acm.org/citation.cfm?id=1287400
    
    頑張って `std::string` と `std::unordered_map` などを使って書くと速くなりそうな気がするが，
    現在のバージョンの Cython はマルチバイト文字周りのラッパー (e.g. `std::wstring` とか?) が
    なかったりと，使うには色々と危なそうな雰囲気があるので，あんまりその辺に手を出したくない．
    今の実装でそこそこの計算時間&消費メモリで動作するのでとりあえずよしとする．
    同じアルゴリズムで pure Python で書いて実行した場合，
    小規模なコーパスではそんなに遅くならない気がしたが，
    大規模なコーパスでどうなるかはちゃんと試してない．
    '''
    
    cdef:
        long size_bucket = int(1/epsilon)
        long len_string = len(string)
        long i, i_bucket = 1
        dict count_dict = dict(), error_dict = dict()

    
    for i in range(1, len_string + 1):
        if i % size_bucket == 0:
            for key, _ in list(count_dict.items()):
                if count_dict[key] + error_dict[key] <= i_bucket:
                    del count_dict[key]
                    del error_dict[key]
                    
            i_bucket += 1
            
        ngram = string[i:(i + n_ngram)]
        
        if ngram in count_dict:
            count_dict[ngram] += 1
        else:
            count_dict[ngram] = 1
            error_dict[ngram] = i_bucket - 1
    
    retval = namedtuple('retval', 'count_dict, error_dict, epsilon, support_threshold, i_bucket, size_bucket, n_char')
    return retval(count_dict, error_dict, epsilon, support_threshold, i_bucket, size_bucket, len_string)

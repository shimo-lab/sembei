import numpy as np
import pandas as pd


def evaluate_by_jawordsim(path_jawordsim, vectors, scatter_plot=False):
    '''Evaluate by Japanese word similarity dataset.

    path_jawordsim
    vectors
    scatter_plot
    '''
    similarities_ja = pd.read_csv(path_jawordsim, skipinitialspace=True)
    
    similarities_predict = []
    for i_row, row in similarities_ja.iterrows():
        if row.word1 in vectors.index and row.word2 in vectors.index:
            similarity = np.dot(vectors.loc[row.word1].values, vectors.loc[row.word2].values)
        else:
            similarity = None
        
        similarities_predict.append(similarity)
        
    similarities_ja.loc[:, 'predict'] = pd.Series(similarities_predict)
    
    correlation = similarities_ja.corr(method='spearman').loc['mean', 'predict']
    coverage = similarities_ja.predict.notnull().mean()
    
    if scatter_plot:
        similarities_ja.plot.scatter('predict', 'mean', figsize=(5, 5))

    return (correlation, coverage, similarities_ja)

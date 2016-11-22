import pandas as pd
from sklearn.decomposition import PCA
import bokeh.plotting as bplt


def get_topn_df(sembei, query_list, topn=10,
                remove_substring=False, print_n_occur=False,
                replace_space=' ', gamma=0.0):
    '''Generate pandas.DataFrame.
    
    Parameters
    ------------------------------
    sembei : sembei.embed.Sembei
    query_list : list of str
    topn : integer (default=10)
    remove_substring : bool
    print_n_occur : bool
    '''

    df_sims = dict()
    vectors = sembei.get_vectors(normalize_vectors=True, gamma=gamma)
    
    for query in query_list:
        if query not in vectors.index:
            print('"{0}" is not in `vectors.index`'.format(query))
            continue
            
        similarities = pd.Series(vectors.values @ vectors.loc[query, :].values,
                                 index=vectors.index)

        similarities_topn = similarities.sort_values(ascending=False).index[1:(topn + 1)]
        colname = query
        
        if print_n_occur:
            n_appear = sembei.G1_diag[sembei.vocabulary.index(query)]
            colname += '({0})'.format(int(n_appear))
            
        df_sims[colname] = similarities_topn

            
    return pd.DataFrame(df_sims).applymap(lambda x: x.replace(' ', replace_space))


def plot_pca_bokeh(vectors_plot, font='IPAexGothic', fontsize='5pt',
                   plot_shape=(800, 800)):
    '''Plot PCA biplot using Bokeh plotting liblary.

    vectors_plot
    font
    fontsize
    plot_shape
    '''
    plot_height, plot_width = plot_shape
    pca = PCA(n_components=2)
    X = pca.fit_transform(vectors_plot)

    fig = bplt.figure(plot_width=plot_width, plot_height=plot_height)
    fig.text(x=X[:, 0].tolist(), y=X[:, 1].tolist(), text=vectors_plot.index,
             text_font_size=fontsize, text_font=font)
    fig.xaxis.axis_label = 'PC1'
    fig.yaxis.axis_label = 'PC2'
    bplt.show(fig)

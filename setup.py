from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


path_cython = ['sembei/embed_inner.pyx', 'sembei/utils/counting.pyx']

setup(
    name='sembei',
    packages=['sembei'],
    ext_modules=cythonize(path_cython),
    include_dirs=[np.get_include()]
)

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


path_cython = ['sembei/embed_inner.pyx', 'sembei/utils/counting.pyx']

setup(
    name='sembei',
    version='0.2',
    packages=['sembei', 'sembei.utils'],
    ext_modules=cythonize(path_cython),
    include_dirs=[np.get_include()]
)

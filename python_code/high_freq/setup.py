# setup.py
# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "hb72.pyx", compiler_directives={"language_level": "3"}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)

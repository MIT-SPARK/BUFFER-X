from setuptools import setup, Extension
import numpy

SOURCES = ["../cpp_utils/cloud/cloud.cpp", "neighbors/neighbors.cpp", "wrapper.cpp"]

eigen_include_dir = "/usr/include/eigen3"
tbb_include_dir = "/usr/include"
tbb_library_dir = "/usr/lib"
tbb_library = "tbb"
kiss_matcher_include_dir = "../cpp_utils/kiss_matcher"

module = Extension(
    name="radius_neighbors",
    sources=SOURCES,
    include_dirs=[
        eigen_include_dir,
        tbb_include_dir,
        kiss_matcher_include_dir,
        numpy.get_include(),
    ],
    extra_compile_args=["-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-O3", "-march=native"],
    extra_link_args=[f"-L{tbb_library_dir}", f"-l{tbb_library}"],
)

setup(
    ext_modules=[module],
)

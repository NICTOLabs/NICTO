"""
NICTO AI - Build C++ Engine Extensions

Build with:
    pip install -e ".[cpp]"

Or standalone:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

# Check for pybind11
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..",
        "Lib", "site-packages", "pybind11", "include"
    )

# Check for torch includes
try:
    import torch
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
except ImportError:
    torch_include = ""
    torch_lib = ""

cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")

nicto_engine = Extension(
    "nicto_ai.engine.nicto_engine",
    sources=[
        os.path.join(cpp_dir, "bindings.cpp"),
    ],
    include_dirs=[
        cpp_dir,
        pybind11_include,
        torch_include,
    ],
    language="c++",
    extra_compile_args=["/std:c++17", "/O2", "/bigobj"] if sys.platform == "win32" else ["-std=c++17", "-O3", "-fPIC"],
    extra_link_args=[],
)

setup(
    name="nicto-engine",
    version="0.1.0",
    description="NICTO AI fast C++ engines",
    ext_modules=[nicto_engine],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

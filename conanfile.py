"""Conan 2 recipe for hnsw-lite: `conan create .` builds and tests the package.

Static library only (the library has no symbol-export macros for a Windows DLL). The CMake
names are those of the project's own package: find_package(hnsw_lite) and the target
hnsw_lite::hnsw_lite, so code using it is the same whichever way the library was obtained.
"""
import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load


class HnswLiteConan(ConanFile):
    name = "hnsw-lite"
    description = ("Vector search: HNSW, flat, IVF, disk-based and sparse indexes, metadata filters, "
                   "hybrid search and durability, in one C++20 library with no dependencies")
    license = "MIT"
    url = "https://github.com/praknig/hnsw_lite"
    homepage = "https://github.com/praknig/hnsw_lite"
    topics = ("vector-search", "hnsw", "nearest-neighbor", "embeddings", "simd")
    package_type = "static-library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}
    # The version lives in the Python package (python/hnsw_lite_lib/__init__.py), read by CMake too.
    exports_sources = ("CMakeLists.txt", "LICENSE", "cmake/*", "include/*", "src/*", "python/hnsw_lite_lib/__init__.py")

    def set_version(self):
        text = load(self, os.path.join(self.recipe_folder, "python", "hnsw_lite_lib", "__init__.py"))
        self.version = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', text, re.M).group(1)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def validate(self):
        check_min_cppstd(self, 20)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        for part in ("TESTS", "BENCHMARKS", "EXAMPLES", "TOOLS", "PYTHON"):
            tc.cache_variables[f"HNSW_LITE_BUILD_{part}"] = False
        tc.cache_variables["HNSW_LITE_INSTALL"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        CMake(self).install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "hnsw_lite")
        self.cpp_info.set_property("cmake_target_name", "hnsw_lite::hnsw_lite")
        self.cpp_info.libs = ["hnsw_lite"]
        self.cpp_info.includedirs = ["include/hnsw_lite", "include"]
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["pthread"]

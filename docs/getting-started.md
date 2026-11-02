# Getting started

This guide covers building hnsw-lite, running its tests and examples, and using it in your own project.

## Requirements

| Tool | Minimum version |
|---|---|
| C++ standard | C++20 |
| GCC (including MinGW / w64devkit on Windows) | 11 |
| Clang | 15 (14 works with GCC 11's standard library, but not with GCC 13's, the default on Ubuntu 24.04) |
| MSVC | Visual Studio 2022 |
| CMake | 3.20 |

Any 64-bit x86 or ARM processor works. The fastest distance code for your CPU (AVX2, AVX-512 or NEON) is chosen automatically when the program starts, so the same build runs everywhere.


## Get the code

```bash
git clone https://github.com/praknig/hnsw_lite.git
cd hnsw_lite
```

## Build and test

Build in **Release** mode for real use and timing. Debug builds turn off optimizations, so they run many times slower (CTest's run of the whole test suite takes about a minute in Release on a 20-thread laptop, many times longer in Debug).

### Linux and macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Windows with GCC (MinGW or w64devkit)

```powershell
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

w64devkit includes Ninja. With a MinGW installation that has no Ninja, use `-G "MinGW Makefiles"` instead.

### Windows with Visual Studio (MSVC)

Open **Developer PowerShell for VS 2022** from the Start menu (an ordinary PowerShell window does not have the compiler set up), then:

```powershell
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

With Visual Studio, programs are placed in `build\Release\`.

### CLion

1. Open the `hnsw_lite` folder (the cloned repository).
2. Add a **Release** profile under Settings → Build, Execution, Deployment → CMake, and select it.
3. Choose **All CTest** in the run configuration list and click **Run**.

### What a successful run looks like

CTest runs the tests as 65 entries (one per group, the heaviest in shards). Each CTest entry runs one group, or one shard of a heavy group, so its log starts with a line such as `12 of 961 tests selected`: that entry's share, not tests left out. Together the 65 entries run all 961 tests; `test_comprehensive` on its own runs them all in one process.

```
100% tests passed, 0 tests failed out of 65
```

The same tests can be run directly, which shows each test by name. See [Testing](testing.md) for everything the test runner can do.

```bash
./build/test_comprehensive                   # all 961 tests
./build/test_comprehensive --group filter    # one group
./build/test_comprehensive --list            # every test name
```

## Run the examples

Three small programs in [examples/](../examples) show the main features. They are built along with the project:

```bash
./build/example_basics        # add, search, update, remove, compact
./build/example_filters       # metadata, filters, predicates, set_metadata
./build/example_batch_range   # batch search and range search
```

On Windows, add `.exe` (for example `.\build\example_basics.exe`).

## Build options

Pass these to the first `cmake` command, for example `-DHNSW_LITE_BUILD_TESTS=OFF`.

| Option | Default | Effect |
|---|---|---|
| `HNSW_LITE_BUILD_TESTS` | ON when hnsw-lite is the main project, OFF when included in another | Builds `test_comprehensive`, `test_adversarial` and `test_races` (and `test_cli` with the tool) and registers them with CTest |
| `HNSW_LITE_BUILD_BENCHMARKS` | Same | Builds `bench_distance`, `bench_search`, `bench_baseline`, `bench_codecs`, `bench_durability` and `bench_locality` |
| `HNSW_LITE_BUILD_EXAMPLES` | Same | Builds the three example programs |
| `HNSW_LITE_BUILD_TOOLS` | Same | Builds the `hnsw-lite` command-line tool (`inspect`, `verify`, `health`, `export`) |
| `HNSW_LITE_BUILD_PYTHON` | OFF | Builds the Python extension module (`hnsw_lite_lib._core`); `pip install .` turns it on for you |
| `HNSW_LITE_SANITIZE` | OFF | Adds AddressSanitizer and UndefinedBehaviorSanitizer with GCC or Clang (ignored with MSVC). They work on Linux and macOS, not with MinGW |
| `HNSW_LITE_INSTALL` | ON when hnsw-lite is the main project (not for the Python build), OFF when included in another | Adds the install rules: `cmake --install` puts the library, its headers (in `include/hnsw_lite/`) and a CMake package in the prefix |

## Use hnsw-lite in your own project

hnsw-lite is a static library: the CMake target `hnsw_lite` (also available as `hnsw_lite::hnsw_lite`). Linking it gives your program the include paths and the thread library it needs.

### Option 1: FetchContent (downloads it for you)

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(hnsw_lite
        GIT_REPOSITORY https://github.com/praknig/hnsw_lite.git
        GIT_TAG master)   # or a release tag, to pin a version
FetchContent_MakeAvailable(hnsw_lite)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE hnsw_lite::hnsw_lite)
```

### Option 2: add_subdirectory (a copy inside your project)

Put the repository in a folder of your project, for example `third_party/hnsw_lite`, then:

```cmake
add_subdirectory(third_party/hnsw_lite)
target_link_libraries(my_app PRIVATE hnsw_lite::hnsw_lite)
```

In both cases only the library is built: tests, benchmarks, examples and the command-line tool are skipped automatically.

### Option 3: an installed package (find_package)

Build and install hnsw-lite once, then use it from any project:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHNSW_LITE_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build
cmake --install build
```

```cmake
find_package(hnsw_lite 0.1 REQUIRED)   # with -DCMAKE_PREFIX_PATH=$HOME/.local if that prefix is not searched
target_link_libraries(my_app PRIVATE hnsw_lite::hnsw_lite)
```

The headers are installed in `include/hnsw_lite/`. Write `#include "hnsw_index.h"`: it works in every setup (FetchContent, add_subdirectory, an installed package, vcpkg, Conan); `#include <hnsw_lite/hnsw_index.h>` works only with an installed package. The package accepts the same minor version only (`0.1` finds 0.1.x, not 0.2): before 1.0, a minor version may change the API.

### Option 4: vcpkg

The repository holds a vcpkg port in `ports/`, used as an overlay port:

```bash
vcpkg install hnsw-lite --overlay-ports=path/to/hnsw_lite/ports
```

then the same `find_package(hnsw_lite CONFIG REQUIRED)` with vcpkg's toolchain file. It builds a static library; it downloads the release tag named by `ports/hnsw-lite/vcpkg.json`'s version. To build an unreleased checkout instead, set `HNSW_LITE_SOURCE_DIR` to it and add `VCPKG_KEEP_ENV_VARS=HNSW_LITE_SOURCE_DIR`.

### Option 5: Conan

`conanfile.py` (Conan 2) packages it from a checkout, with C++20:

```bash
conan create path/to/hnsw_lite -s compiler.cppstd=20
```

then `hnsw-lite/0.1.0` in your `conanfile.txt` or `conanfile.py` requirements, and the same `find_package(hnsw_lite)` and target through Conan's `CMakeDeps`. Static library only; `fPIC` is an option.

### Your first program

```cpp
// main.cpp
#include <cstdio>
#include <vector>

#include "hnsw_index.h"

int main() {
    vecdb::HnswIndex index(3, vecdb::Metric::L2);
    index.add(1, std::vector<float>{0.0f, 0.0f, 0.0f});
    index.add(2, std::vector<float>{1.0f, 1.0f, 1.0f});
    index.add(3, std::vector<float>{5.0f, 5.0f, 5.0f});

    for (const auto& r : index.search(std::vector<float>{0.9f, 1.0f, 1.1f}, 2))
        std::printf("id %llu  distance %.2f\n", (unsigned long long)r.id, r.distance);
}
```

Output (L2 distances are squared):

```
id 2  distance 0.02
id 1  distance 3.02
```

All public names live in the `vecdb` namespace. Include `hnsw_index.h` for `HnswIndex` or `flat_index.h` for `FlatIndex`; both bring in metadata, filters and search options. Continue with the [User guide](user-guide.md).

## Python

`pip install hnsw-lite-lib` installs a prebuilt wheel. `pip install .` from the repository root builds the library into the Python package `hnsw_lite_lib` (pip fetches pybind11 and scikit-build-core; a C++20 compiler and CMake are needed). `pip install ".[test]"` and `python -m pytest` run its tests. See the [Python guide](python.md).

## Troubleshooting

| Problem | Solution |
|---|---|
| CMake keeps using the wrong compiler | Delete the `build` folder. CMake remembers the compiler it first chose there. |
| `cl` or MSVC not found | Use **Developer PowerShell for VS 2022**, not a plain PowerShell window. |
| Everything is very slow | You are probably running a Debug build. Rebuild with `-DCMAKE_BUILD_TYPE=Release` (or `--config Release` with Visual Studio). |
| `std::span` or other C++20 errors in your project | Set `CMAKE_CXX_STANDARD` to 20 in your own `CMakeLists.txt`. |
| A test reports `SKIP` | Expected in two cases: tests that need another operating system's features (on Windows, one that reads memory through Linux's `/proc` and one that needs POSIX `fork()`, so 959 passed and 2 skipped; on macOS, the `/proc` one only); and under the sanitizers, the out-of-memory tests, which need a custom memory allocator that the sanitizers replace. |
| Your CPU lacks AVX2 or AVX-512 | Nothing to do: a slower version is chosen automatically, with identical results. The test output's first line shows which one (`kernel: ...`). |
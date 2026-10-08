# Getting started

This guide covers building hnsw-lite, running its tests and examples, and using it in your own project.

## Requirements

| Tool | Minimum version |
|---|---|
| C++ standard | C++20 |
| GCC (including MinGW / w64devkit on Windows) | 11 |
| Clang | 14 |
| MSVC | Visual Studio 2022 |
| CMake | 3.20 |

Any 64-bit x86 or ARM processor works. The fastest distance code for your CPU (AVX2, AVX-512 or NEON) is chosen automatically when the program starts, so the same build runs everywhere.

## Get the code

```bash
git clone https://github.com/praknig/hnsw_lite.git
cd hnsw_lite
```

## Build and test

Build in **Release** mode for real use and timing. Debug builds turn off optimizations, so they run many times slower (the test suite takes minutes instead of about 15 seconds).

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

### Windows with Visual Studio (MSVC)

Open **Developer PowerShell for VS 2022** from the Start menu (an ordinary PowerShell window does not have the compiler set up), then:

```powershell
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

With Visual Studio, programs are placed in `build\Release\`.

### CLion

1. Open the `hnsw_lite` folder.
2. Add a **Release** profile under Settings → Build, Execution, Deployment → CMake, and select it.
3. Choose **All CTest** in the run configuration list and click **Run**.

### What a successful run looks like

CTest runs the tests as 18 groups:

```
100% tests passed, 0 tests failed out of 18
```

The same tests can be run directly, which shows each test by name. See [Testing](testing.md) for everything the test runner can do.

```bash
./build/test_comprehensive                   # all 445 tests
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
| `HNSW_LITE_BUILD_TESTS` | ON when hnsw-lite is the main project, OFF when included in another | Builds `test_comprehensive` and registers it with CTest |
| `HNSW_LITE_BUILD_BENCHMARKS` | Same | Builds `bench_distance` and `bench_search` |
| `HNSW_LITE_BUILD_EXAMPLES` | Same | Builds the three example programs |
| `HNSW_LITE_SANITIZE` | OFF | Adds AddressSanitizer and UndefinedBehaviorSanitizer (GCC and Clang on Linux and macOS only) |

## Use hnsw-lite in your own project

hnsw-lite is a static library named `hnsw_lite` (also available as `hnsw_lite::hnsw_lite`). Linking it gives your program the include paths and the thread library it needs.

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

In both cases only the library is built: tests, benchmarks and examples are skipped automatically.

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

## Troubleshooting

| Problem | Solution |
|---|---|
| CMake keeps using the wrong compiler | Delete the `build` folder. CMake remembers the compiler it first chose there. |
| `cl` or MSVC not found | Use **Developer PowerShell for VS 2022**, not a plain PowerShell window. |
| Everything is very slow | You are probably running a Debug build. Rebuild with `-DCMAKE_BUILD_TYPE=Release` (or `--config Release` with Visual Studio). |
| `std::span` or other C++20 errors in your project | Set `CMAKE_CXX_STANDARD` to 20 in your own `CMakeLists.txt`. |
| A test group reports `SKIP` | Expected under the sanitizers: out-of-memory tests need a custom memory allocator that sanitizers replace. |
| Your CPU lacks AVX2 or AVX-512 | Nothing to do: a slower version is chosen automatically, with identical results. The test output's first line shows which one (`kernel: ...`). |
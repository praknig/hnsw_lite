# hnsw-lite's vcpkg port. Use it from this repository as an overlay port:
#   vcpkg install hnsw-lite --overlay-ports=<repository>/ports
# For each release, set "version" in vcpkg.json and SHA512 below (with SHA512 0, vcpkg's
# first build reports the right value). To build an unreleased checkout instead, set
# HNSW_LITE_SOURCE_DIR to it and pass it through: VCPKG_KEEP_ENV_VARS=HNSW_LITE_SOURCE_DIR.

# A static library: it has no symbol-export macros for a Windows DLL.
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(DEFINED ENV{HNSW_LITE_SOURCE_DIR})
    set(SOURCE_PATH "$ENV{HNSW_LITE_SOURCE_DIR}")
    message(STATUS "hnsw-lite: building the local checkout ${SOURCE_PATH}")
else()
    vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REPO praknig/hnsw_lite
        REF "v${VERSION}"
        SHA512 0
        HEAD_REF master
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DHNSW_LITE_INSTALL=ON
        -DHNSW_LITE_BUILD_TESTS=OFF
        -DHNSW_LITE_BUILD_BENCHMARKS=OFF
        -DHNSW_LITE_BUILD_EXAMPLES=OFF
        -DHNSW_LITE_BUILD_TOOLS=OFF
        -DHNSW_LITE_BUILD_PYTHON=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME hnsw_lite CONFIG_PATH lib/cmake/hnsw_lite)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

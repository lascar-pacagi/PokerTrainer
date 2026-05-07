cmake_minimum_required(VERSION 3.20)
project(pokertrainer_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─── Build options ──────────────────────────────────────────────────────────
option(PT_BUILD_TESTS       "Build Catch2 unit tests"      ON)
option(PT_BUILD_PY_BINDINGS "Build pybind11 Python module" ON)
option(PT_BUILD_FFI         "Build C ABI for Flutter FFI"  ON)
option(PT_BUILD_INFER       "Build LibTorch C++ inference (requires torch)" OFF)
option(PT_BUILD_VALIDATION  "Build independent BR validator for pt-solver" OFF)
option(PT_ENABLE_AVX2       "Enable AVX2 fast path"        ON)

# ─── LibTorch detection (runs before FFI so both targets can share it) ──────
# We deliberately bypass find_package(Torch) because torch's CMake insists on
# enabling CUDA language support even for a CPU-only link, and the system
# nvcc may not match the torch CUDA build. Instead, link libtorch_cpu + libc10
# + libtorch directly from the Python torch package.
if(PT_BUILD_INFER)
    if(NOT PT_BUILD_PY_BINDINGS)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
    else()
        find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import os, torch; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE PT_TORCH_DIR
        RESULT_VARIABLE _PT_TORCH_RC
    )
    if(NOT _PT_TORCH_RC EQUAL 0 OR NOT EXISTS "${PT_TORCH_DIR}/lib/libtorch_cpu.so")
        message(FATAL_ERROR
            "PT_BUILD_INFER=ON but LibTorch not found via Python3 "
            "(expected ${PT_TORCH_DIR}/lib/libtorch_cpu.so).")
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import torch; print(int(torch.compiled_with_cxx11_abi()), end='')"
        OUTPUT_VARIABLE PT_TORCH_CXX11_ABI
    )
    message(STATUS "LibTorch: ${PT_TORCH_DIR}  (cxx11-abi=${PT_TORCH_CXX11_ABI})")
endif()

# ─── Core library ───────────────────────────────────────────────────────────
add_library(pt_core STATIC
    src/card.cpp
    src/action.cpp
    src/hand_eval.cpp
    src/game_hu.cpp
    src/encoder.cpp
    src/env.cpp
)
target_include_directories(pt_core PUBLIC src)
set_target_properties(pt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(pt_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
)
if(PT_ENABLE_AVX2)
    target_compile_options(pt_core PUBLIC -mavx2 -mbmi2)
    target_compile_definitions(pt_core PUBLIC PT_AVX2=1)
endif()

# ─── Tests ──────────────────────────────────────────────────────────────────
if(PT_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
    enable_testing()

    add_executable(pt_tests
        tests/test_card.cpp
        tests/test_hand_eval.cpp
        tests/test_game_hu.cpp
        tests/test_encoder.cpp
        tests/test_env.cpp
    )
    target_link_libraries(pt_tests PRIVATE pt_core Catch2::Catch2WithMain)
    target_compile_definitions(pt_tests PRIVATE
        PT_TABLES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/poker_tables.bin"
    )

    include(CTest)
    include(Catch)
    catch_discover_tests(pt_tests)
endif()

# ─── Python bindings (pybind11) ─────────────────────────────────────────────
if(PT_BUILD_PY_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(pokertrainer_engine bindings/py_env.cpp)
    target_link_libraries(pokertrainer_engine PRIVATE pt_core)
endif()

# ─── Flutter FFI (C ABI) ────────────────────────────────────────────────────
if(PT_BUILD_FFI)
    add_library(pokertrainer_ffi SHARED bindings/ffi.cpp)
    target_link_libraries(pokertrainer_ffi PRIVATE pt_core)
    set_target_properties(pokertrainer_ffi PROPERTIES
        OUTPUT_NAME "pokertrainer"
        POSITION_INDEPENDENT_CODE ON
    )
    # If inference is also enabled, wire the torch-backed model_* symbols in.
    # The ffi library then carries an rpath to the torch shared libs so
    # hosts (Flutter, Python ctypes) can load it without LD_LIBRARY_PATH.
    if(PT_BUILD_INFER)
        target_link_libraries(pokertrainer_ffi PRIVATE pt_infer)
        target_compile_definitions(pokertrainer_ffi PRIVATE PT_FFI_INFER=1)
        set_target_properties(pokertrainer_ffi PROPERTIES
            INSTALL_RPATH "${PT_TORCH_DIR}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
            BUILD_RPATH "${PT_TORCH_DIR}/lib"
        )
    endif()
endif()

# ─── LibTorch inference target (uses variables from the detection block) ───
if(PT_BUILD_INFER)
    add_library(pt_infer STATIC src/inference.cpp)
    target_include_directories(pt_infer PUBLIC
        src
        "${PT_TORCH_DIR}/include"
        "${PT_TORCH_DIR}/include/torch/csrc/api/include"
    )
    target_link_directories(pt_infer PUBLIC "${PT_TORCH_DIR}/lib")
    target_link_libraries(pt_infer PUBLIC pt_core torch_cpu c10 torch)
    target_compile_definitions(pt_infer PUBLIC
        "_GLIBCXX_USE_CXX11_ABI=${PT_TORCH_CXX11_ABI}")
    set_target_properties(pt_infer PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "${PT_TORCH_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    add_executable(play_match_cpp tools/play_match_cpp.cpp)
    target_link_libraries(play_match_cpp PRIVATE pt_infer)
    set_target_properties(play_match_cpp PROPERTIES
        INSTALL_RPATH "${PT_TORCH_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${PT_TORCH_DIR}/lib"
    )
endif()

# ─── Independent BR validator for pt-solver ─────────────────────────────────
if(PT_BUILD_VALIDATION)
    add_subdirectory(validation)
endif()

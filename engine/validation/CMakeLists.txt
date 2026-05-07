# Independent best-response validator for pt-solver dumps.
#
# Standalone target. Reuses pt_core (for hand-eval + Card types) and
# vendors nlohmann/json header-only. Built when PT_BUILD_VALIDATION is ON
# (default OFF; enable from the engine root with -DPT_BUILD_VALIDATION=ON).

add_library(pt_validation STATIC
    scenario.cpp
    best_response.cpp
)
target_include_directories(pt_validation PUBLIC .)
target_include_directories(pt_validation SYSTEM PUBLIC third_party)
target_link_libraries(pt_validation PUBLIC pt_core)
target_compile_options(pt_validation PRIVATE -O2 -Wall -Wextra)

add_executable(pt_validate main.cpp)
target_link_libraries(pt_validate PRIVATE pt_validation)
target_compile_options(pt_validate PRIVATE -O2 -Wall -Wextra)

# Tests.
if(PT_BUILD_TESTS AND TARGET Catch2::Catch2WithMain)
    add_executable(pt_validation_tests
        tests/test_tiny.cpp
        tests/test_blockers.cpp
        tests/test_paranoid.cpp
    )
    target_link_libraries(pt_validation_tests PRIVATE
        pt_validation Catch2::Catch2WithMain)
    target_compile_definitions(pt_validation_tests PRIVATE
        PT_TABLES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../data/poker_tables.bin"
        PT_SOLVER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../validation/target/release/pt-solver"
        PT_VALIDATION_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

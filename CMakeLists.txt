cmake_minimum_required(VERSION 3.16)
project(mecanum_mpc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

option(MPC_USE_NEON "Use custom NEON kernels for online linear algebra" OFF)
option(MPC_ENABLE_RERUN "Enable Rerun visualization for tests" OFF)

# Find BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(mecanum_mpc STATIC
    src/mecanum_model.cpp
    src/discretizer.cpp
    src/condensing.cpp
    src/cholesky.cpp
    src/box_qp_solver.cpp
    src/mpc_offline.cpp
    src/mpc_online.cpp
    src/blas_dispatch.cpp
)
target_link_libraries(mecanum_mpc PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(mecanum_mpc PUBLIC include/)

if(MPC_USE_NEON)
    target_sources(mecanum_mpc PRIVATE src/neon_kernels.cpp)
    target_compile_definitions(mecanum_mpc PUBLIC MPC_USE_NEON=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # AArch64: NEON always available
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        target_compile_options(mecanum_mpc PRIVATE -mfpu=neon -mfloat-abi=hard)
    endif()
endif()

# Tests
enable_testing()

# Helper function for tests
function(add_mpc_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE mecanum_mpc)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_mpc_test(test_discretizer)
add_mpc_test(test_condensing)
add_mpc_test(test_cholesky)
add_mpc_test(test_box_qp)
add_mpc_test(test_mpc_roundtrip)
add_mpc_test(test_timing)

# Rerun visualization test (optional, not part of main library)
if(MPC_ENABLE_RERUN)
    include(FetchContent)

    FetchContent_Declare(
        rerun_sdk
        URL https://github.com/rerun-io/rerun/releases/download/0.25.1/rerun_cpp_sdk.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(rerun_sdk nlohmann_json)

    add_executable(test_trajopt_rerun tests/test_trajopt_rerun.cpp)
    target_link_libraries(test_trajopt_rerun PRIVATE mecanum_mpc rerun_sdk nlohmann_json::nlohmann_json)
endif()

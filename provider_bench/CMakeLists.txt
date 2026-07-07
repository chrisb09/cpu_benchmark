cmake_minimum_required(VERSION 3.16)
project(provider_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(PROVIDER_BENCH_WITH_AIX "Build provider_bench with AIxelerator support" ON)

set(CPP_MODULE_DIR "/hpcwork/ro092286/smartsim/CPP-ML-Interface")
set(SMARTREDIS_DIR "${CPP_MODULE_DIR}/extern/SmartRedis")
set(SMARTSIM_PYTHON "${CPP_MODULE_DIR}/extern/python/smartsim_cuda-12/bin/python"
    CACHE FILEPATH "Python interpreter with clang module available")
set(PHYDLL_BUILD_DIR "${CPP_MODULE_DIR}/extern/phydll/build")

if(NOT EXISTS "${SMARTSIM_PYTHON}")
	message(FATAL_ERROR "SMARTSIM_PYTHON not found: ${SMARTSIM_PYTHON}")
endif()

set(GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.hpp")
set(TEMP_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.tmp.hpp")

file(GLOB_RECURSE CPP_MODULE_HEADERS
	"${CPP_MODULE_DIR}/include/application/*.hpp"
	"${CPP_MODULE_DIR}/include/behavior/*.hpp"
	"${CPP_MODULE_DIR}/include/normalization/*.hpp"
	"${CPP_MODULE_DIR}/include/provider/*.hpp"
)
set(ALL_HEADERS ${CPP_MODULE_HEADERS})
list(REMOVE_DUPLICATES ALL_HEADERS)

set(BASE_CLASSES "MLCouplingProvider,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication")
set(CLANG_ARGS "-std=c++17 -I${CPP_MODULE_DIR}/include -I${SMARTREDIS_DIR}/include -I${SMARTREDIS_DIR}/install/include -I${CPP_MODULE_DIR}/build/_deps/tomlplusplus-src/include -I${PHYDLL_BUILD_DIR}/include -xc++")

add_custom_command(
	OUTPUT "${GENERATED_REGISTRY}"
	COMMAND ${CMAKE_COMMAND} -E env "CLANG_ARGS=${CLANG_ARGS}"
			"${SMARTSIM_PYTHON}" "${CPP_MODULE_DIR}/scripts/generate_registry.py"
			"${TEMP_GENERATED_REGISTRY}" "${BASE_CLASSES}" ${ALL_HEADERS}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${GENERATED_REGISTRY}"
	DEPENDS ${ALL_HEADERS} "${CPP_MODULE_DIR}/scripts/generate_registry.py"
	WORKING_DIRECTORY "${CPP_MODULE_DIR}"
	COMMENT "Generating registry for provider_bench..."
)
add_custom_target(benchmark_registry DEPENDS "${GENERATED_REGISTRY}")

add_executable(benchmark_solver benchmark_solver.cpp)
add_dependencies(benchmark_solver benchmark_registry)

set(CPPML_DISABLE_REGISTRY_GEN ON CACHE BOOL "Disable CPP-ML-Interface registry generation" FORCE)
set(WITH_PHYDLL ON CACHE BOOL "Enable PhyDLL provider support through CPP-ML-Interface" FORCE)
set(WITH_AIX ${PROVIDER_BENCH_WITH_AIX} CACHE BOOL "Enable AIxelerator support through CPP-ML-Interface" FORCE)

add_subdirectory("${CPP_MODULE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build" EXCLUDE_FROM_ALL)

# Make cpp_ml_interface_library depend on our registry so that c_api.cpp can find it
add_dependencies(cpp_ml_interface_library benchmark_registry)

target_include_directories(benchmark_solver PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}"
	"${CPP_MODULE_DIR}/include"
	"${SMARTREDIS_DIR}/include"
	"${SMARTREDIS_DIR}/install/include"
	"${CPP_MODULE_DIR}/build/_deps/tomlplusplus-src/include"
	"${PHYDLL_BUILD_DIR}/include"
)

# Crucial: the cpp_ml_interface_library needs to find our generated_registry.hpp too
target_include_directories(cpp_ml_interface_library PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(benchmark_solver PRIVATE cpp_ml_interface_headers cpp_ml_interface_library /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/extras/CUPTI/lib64/libcupti.so.12)
target_link_options(benchmark_solver PRIVATE "-Wl,--unresolved-symbols=ignore-all")

find_package(MPI)
if(MPI_FOUND)
    add_compile_definitions(MPI_FOUND)
endif()

target_compile_definitions(benchmark_solver PRIVATE SMARTSIM_ENABLED WITH_SMARTSIM WITH_PHYDLL)
if(PROVIDER_BENCH_WITH_AIX)
    target_compile_definitions(benchmark_solver PRIVATE WITH_AIX)
endif()

if(EXISTS "${SMARTREDIS_DIR}/build/Release/libsmartredis.so")
	set(SMARTREDIS_LIB_DIR "${SMARTREDIS_DIR}/build/Release")
else()
	set(SMARTREDIS_LIB_DIR "${SMARTREDIS_DIR}/install/lib64")
endif()

target_link_directories(benchmark_solver PRIVATE "${SMARTREDIS_LIB_DIR}")
target_link_directories(benchmark_solver PRIVATE "${PHYDLL_BUILD_DIR}/lib")

target_link_libraries(benchmark_solver PRIVATE smartredis phydll mpi)

cmake_minimum_required(VERSION 3.20)

project(bench_dolfinx C CXX)

include(GNUInstallDirs)

# Default to an optimised build (-O3 -DNDEBUG). Use
# -DCMAKE_BUILD_TYPE=Debug for a debug build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # https://cmake.org/cmake/help/latest/policy/CMP0144.html
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW) # Boost CONFIG mode
endif()

# C++20, no language extensions
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(HIP_ARCH "none" CACHE STRING "Set the AMD GPU target")
set(CUDA_ARCH "none" CACHE STRING "Set the NVIDIA GPU architecture")

if(NOT HIP_ARCH STREQUAL "none")
  set(amd ON)
endif()

if(NOT CUDA_ARCH STREQUAL "none")
  set(nvidia ON)
endif()

if(nvidia AND amd)
  message(
    FATAL_ERROR
      "You cannot select both AMD and NVIDIA GPU builds. Only set either CUDA_ARCH or HIP_ARCH."
  )
elseif(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs with arch ${CUDA_ARCH}")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs with target ${HIP_ARCH}")
else()
  message(STATUS "No GPU arch selected. Building for CPU.")
endif()

if(amd)
  # autodetect = native MI250X = gfx90a MI300X = gfx942 Radeon7900 = gfx1100
  set(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH})
  set(AMDGPU_TARGETS ${HIP_ARCH})
  enable_language(HIP)
elseif(nvidia)
  # See
  # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # A100 = 80 RTX6000Ada = 89 GH200 = 90
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  enable_language(CUDA)
endif()

# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)

if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
find_package(Boost 1.70 REQUIRED program_options json)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Compile UFL code
add_custom_command(
  OUTPUT poisson64.c
  COMMAND ffcx --scalar_type float64 --sum_factorization
          ${CMAKE_CURRENT_SOURCE_DIR}/poisson64.py
  VERBATIM
  DEPENDS poisson64.py
  COMMENT "Compile poisson64.py [f64] using FFCx")

add_custom_command(
  OUTPUT poisson32.c
  COMMAND ffcx --scalar_type float32 --sum_factorization
          ${CMAKE_CURRENT_SOURCE_DIR}/poisson32.py
  VERBATIM
  DEPENDS poisson32.py
  COMMENT "Compile poisson32.py [f32] using FFCx")


if(amd)
  find_package(rocthrust REQUIRED CONFIG)
endif()

add_executable(${PROJECT_NAME} main.cpp forms.cpp laplacian_solver.cpp mesh.cpp util.cpp poisson64.c poisson32.c)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} dolfinx Basix::basix Boost::program_options Boost::json)

# Compiler options (optimisation level and NDEBUG come from the build
# type)
target_compile_options(
  ${PROJECT_NAME}
  PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-ffast-math;-Wall>"
          "$<$<COMPILE_LANGUAGE:HIP>:-ffast-math;-Wall;-munsafe-fp-atomics>"
          "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;--extended-lambda;--expt-relaxed-constexpr>")
if(CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18)
  # See
  # https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fclang-abi-compat=17>)
endif()

if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)
  set_source_files_properties(main.cpp forms.cpp laplacian_solver.cpp mesh.cpp util.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
  set_source_files_properties(main.cpp forms.cpp laplacian_solver.cpp mesh.cpp util.cpp PROPERTIES LANGUAGE CUDA)
endif()

# On Cray systems the compiler wrappers (cc, CC) add the MPI include
# path, but the HIP and CUDA compilers are not wrapped. Add the Cray
# MPICH include directory (CRAY_MPICH_DIR is set by the cray-mpich
# module).
if((amd OR nvidia) AND DEFINED ENV{CRAY_MPICH_DIR})
  message(STATUS "Adding Cray MPICH include directory: $ENV{CRAY_MPICH_DIR}/include")
  target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{CRAY_MPICH_DIR}/include")
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

cmake_minimum_required(VERSION 3.20)

project(bench_dolfinx C CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # https://cmake.org/cmake/help/latest/policy/CMP0144.html
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW) # Boost CONFIG mode
endif()

# No C++ language extensions
set(CMAKE_CXX_EXTENSIONS OFF)

set(HIP_ARCH "none" CACHE STRING "Set the AMD GPU target")
set(CUDA_ARCH "none" CACHE STRING "Set the NVIDIA GPU architecture")

if(NOT HIP_ARCH STREQUAL "none")
  set(amd ON)
endif()

if(NOT CUDA_ARCH STREQUAL "none")
  set(nvidia ON)
endif()

if(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs with arch ${CUDA_ARCH}")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs with target ${HIP_ARCH}")
elseif(nvidia AND amd)
  message(
    FATAL_ERROR
      "You cannot select both AMD and NVIDIA GPU builds. Only set either CUDA_ARCH or HIP_ARCH."
  )
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
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

set(CMAKE_CXX_FLAGS "-O3 -ffast-math ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-O3 -ffast-math ${CMAKE_C_FLAGS} -DNDEBUG -Wall")

if(CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18)
  # See
  # https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fclang-abi-compat=17")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -O3 -ffast-math -DNDEBUG -munsafe-fp-atomics")

if(DEFINED ENV{CRAYPE_VERSION})
  message("Detected CrayPE - adding MPI flags")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math -I/opt/cray/pe/mpich/8.1.32/ofi/gnu/12.3/include")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 -O3 -Wall -DNDEBUG --extended-lambda --expt-relaxed-constexpr")

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

# if(NOT amd AND NOT nvidia)
# CPU-only source files
# target_sources(${PROJECT_NAME} PUBLIC geometry_cpu.cpp)
# endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} dolfinx Basix::basix Boost::program_options Boost::json)

if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
  add_definitions(-DUSE_HIP=1)
  set_source_files_properties(main.cpp forms.cpp laplacian_solver.cpp mesh.cpp util.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1)
  set_source_files_properties(main.cpp forms.cpp laplacian_solver.cpp mesh.cpp util.cpp PROPERTIES LANGUAGE CUDA)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

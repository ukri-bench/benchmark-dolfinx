// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "laplacian_solver.hpp"
#include "laplacian.hpp"
#include <basix/quadrature.h>
#include <dolfinx/la/MatrixCSR.h>

using namespace benchdolfinx;

#if defined(USE_CUDA) || defined(USE_HIP)

#include "csr.hpp"
#include "vector.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace
{
/// @brief pack data before MPI (neighbor) all-to-all operation
/// @tparam T Scalar data type
/// @param N Number of entries in indices
/// @param indices Indices of input data to be packed
/// @param in Input data to be sent: from owned region, for forward scatter, or
/// from ghost region for reverse scatter.
/// @param out Output data packed into blocks for each receiving MPI process
template <typename T>
static __global__ void pack(const int N,
                            const std::int32_t* __restrict__ indices,
                            const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    out[gid] = in[indices[gid]];
  }
}

/// @brief unpack data after MPI all-to-all operation
/// @tparam T Scalar data type
/// @param N Number of entries in indices
/// @param indices Indices of output data to be unpacked into
/// @param in Input data packed in blocks received from each MPI process
/// @param out Output data: to ghost region if forward scatter, or to owned
/// region for reverse scatter.
/// @note Overwrites values if multiple are received with the same index, should
/// only be used for forward scatter to ghost region.
template <typename T>
static __global__ void unpack(const int N,
                              const std::int32_t* __restrict__ indices,
                              const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    out[indices[gid]] = in[gid];
  }
}
} // namespace

//----------------------------------------------------------------------------
template <typename T>
BenchmarkResults benchdolfinx::laplace_action_gpu(
    const dolfinx::fem::Form<T>& a, const dolfinx::fem::Form<T>& L,
    const dolfinx::fem::DirichletBC<T>& bc, int degree, int qmode, T kappa,
    int nreps, bool use_gauss, bool matrix_comparison)
{
  auto V = a.function_spaces()[0];

  // Define vectors
  using DeviceVector = benchdolfinx::Vector<T>;

  // Input vector
  auto map = V->dofmap()->index_map;
  spdlog::info("Create device vector u");

  DeviceVector u(map, 1);
  thrust::fill(u.array().begin(), u.array().end(), 0);

  // Output vector
  spdlog::info("Create device vector y");
  DeviceVector y(map, 1);
  thrust::fill(y.array().begin(), y.array().end(), 0);

  // Create matrix free operator
  spdlog::info("Create MatFreeLaplacian");
  dolfinx::common::Timer op_create_timer("% Create matfree operator");

  basix::quadrature::type quad_type
      = use_gauss ? basix::quadrature::type::gauss_jacobi
                  : basix::quadrature::type::gll;

  MatFreeLaplacianGPU<T> op(*V, bc, degree, qmode, kappa, quad_type);

  op_create_timer.stop();

  dolfinx::la::Vector<T> b(map, 1);
  dolfinx::fem::assemble_vector(b.array(), L);
  dolfinx::fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1.0));
  b.scatter_rev(std::plus<T>());
  bc.set(b.array(), std::nullopt);

  // u.copy_from_host(b); // Copy data from host vector to device vector

  // Copy data from host vector to device vector. Copies only local data.
  thrust::copy_n(b.array().begin(), u.index_map()->size_local(),
                 u.array().begin());

  u.scatter_fwd_begin(get_pack_fn<T>(512),
                      [](auto&& x) { return x.data().get(); });
  u.scatter_fwd_end(get_unpack_fn<T>(512, 1));

  BenchmarkResults b_results = {0};

  // Matrix free
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nreps; ++i)
    op.apply(u, y);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;

  T unorm = benchdolfinx::norm(u);
  T ynorm = benchdolfinx::norm(y);

  b_results.unorm = unorm;
  b_results.ynorm = ynorm;

  int rank = dolfinx::MPI::rank(V->mesh()->comm());
  if (rank == 0)
  {
    std::int64_t ndofs_global = V->dofmap()->index_map->size_global();
    b_results.mat_free_time = duration.count();
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    std::cout << "Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;

    std::cout << "Norm of u = " << unorm << std::endl;
    std::cout << "Norm of y = " << ynorm << std::endl;
  }

  if (matrix_comparison)
  {

    // Assemble on CPU and copy to GPU
    std::unique_ptr<benchdolfinx::MatrixOperator<T>> mat_op;
    {

      if (a.rank() != 2)
        throw std::runtime_error("Form should have rank be 2.");

      auto V = a.function_spaces()[0];
      dolfinx::la::SparsityPattern pattern
          = dolfinx::fem::create_sparsity_pattern(a);
      pattern.finalize();

      std::cout << "NNZ = " << pattern.num_nonzeros() << "\n";

      // Note: currently using CPU assembly, the 32-bit limit for row_ptr is
      // unlikely to be reached as creating such a large matrix is prohibitively
      // slow.
      if (pattern.num_nonzeros() >= std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(
            "Too many matrix entries, need 64-bit row_ptr.");

      dolfinx::common::Timer m1("% Create CPU MatrixCSR");
      dolfinx::la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>,
                             std::vector<std::int32_t>>
          A(pattern);
      m1.stop();
      m1.flush();

      dolfinx::common::Timer m2("% Assemble CPU MatrixCSR");
      dolfinx::fem::assemble_matrix(A.mat_add_values(), a, {bc});
      A.scatter_rev();
      dolfinx::fem::set_diagonal<T>(A.mat_set_values(), *V, {bc}, T(1.0));
      m2.stop();
      m2.flush();

      dolfinx::common::Timer m3("% Copy to GPU MatrixCSR");
      mat_op = std::make_unique<benchdolfinx::MatrixOperator<T>>(A);
      m3.stop();
      m3.flush();
    }

    DeviceVector z(map, 1);
    thrust::fill(z.array().begin(), z.array().end(), 0);

    dolfinx::common::Timer mtimer("% CSR Matvec");
    for (int i = 0; i < nreps; ++i)
      mat_op->apply(u, z);
    mtimer.stop();
    mtimer.flush();

    T unorm = benchdolfinx::norm(u);
    T znorm = benchdolfinx::norm(z);
    // Compute error
    DeviceVector e(map, 1);
    benchdolfinx::axpy(e, T{-1}, y, z);
    T enorm = benchdolfinx::norm(e);

    b_results.znorm = znorm;
    b_results.enorm = enorm;

    if (rank == 0)
    {
      std::cout << "Norm of u = " << unorm << std::endl;
      std::cout << "Norm of z = " << znorm << std::endl;
      std::cout << "Norm of error = " << enorm << std::endl;
      std::cout << "Relative norm of error = " << enorm / znorm << std::endl;
    }
  }

  return b_results;
}
//----------------------------------------------------------------------------
/// @cond protect from doxygen
template benchdolfinx::BenchmarkResults
benchdolfinx::laplace_action_gpu<double>(
    const dolfinx::fem::Form<double>&, const dolfinx::fem::Form<double>&,
    const dolfinx::fem::DirichletBC<double>&, int, int, double, int, bool,
    bool);

template benchdolfinx::BenchmarkResults benchdolfinx::laplace_action_gpu<float>(
    const dolfinx::fem::Form<float>&, const dolfinx::fem::Form<float>&,
    const dolfinx::fem::DirichletBC<float>&, int, int, float, int, bool, bool);
/// @endcond
#endif
//----------------------------------------------------------------------------
template <typename T>
BenchmarkResults benchdolfinx::laplace_action_cpu(
    const dolfinx::fem::Form<T>& a, const dolfinx::fem::Form<T>& L,
    const dolfinx::fem::DirichletBC<T>& bc, int degree, int qmode, T kappa,
    int nreps, bool use_gauss, bool matrix_comparison)
{
  auto V = a.function_spaces()[0];

  // Input vector
  auto map = V->dofmap()->index_map;

  spdlog::info("Create vector u");
  dolfinx::la::Vector<T> u(map, 1);
  std::ranges::fill(u.array(), T{0});

  // Output vector
  spdlog::info("Create vector y");
  dolfinx::la::Vector<T> y(map, 1);
  std::ranges::fill(u.array(), T{0});

  // Create matrix free operator
  spdlog::info("Create MatFreeLaplacian");
  dolfinx::common::Timer op_create_timer("% Create matfree operator");

  basix::quadrature::type quad_type
      = use_gauss ? basix::quadrature::type::gauss_jacobi
                  : basix::quadrature::type::gll;

  MatFreeLaplacianCPU<T> op(*V, bc, degree, qmode, kappa, quad_type);

  op_create_timer.stop();

  dolfinx::fem::assemble_vector(u.array(), L);
  dolfinx::fem::apply_lifting(u.array(), {a}, {{bc}}, {}, T(1.0));
  u.scatter_rev(std::plus<T>());
  bc.set(u.array(), std::nullopt);

  BenchmarkResults b_results = {0};
  // Matrix free
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nreps; ++i)
    op.apply(u, y);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;

  T unorm = dolfinx::la::norm(u);
  T ynorm = dolfinx::la::norm(y);

  b_results.unorm = unorm;
  b_results.ynorm = ynorm;

  int rank = dolfinx::MPI::rank(V->mesh()->comm());
  if (rank == 0)
  {
    b_results.mat_free_time = duration.count();

    std::int64_t ndofs_global = V->dofmap()->index_map->size_global();
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    std::cout << "Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;

    std::cout << "Norm of u = " << unorm << std::endl;
    std::cout << "Norm of y = " << ynorm << std::endl;
  }

  if (matrix_comparison)
  {
    dolfinx::la::Vector<T> z(map, 1);
    std::fill(z.array().begin(), z.array().end(), 0);

    dolfinx::la::SparsityPattern sp = dolfinx::fem::create_sparsity_pattern(a);
    sp.finalize();
    dolfinx::la::MatrixCSR<T> mat_op(sp);
    dolfinx::fem::assemble_matrix(mat_op.mat_add_values(), a, {bc});
    mat_op.scatter_rev();
    dolfinx::fem::set_diagonal<T>(mat_op.mat_set_values(), *V, {bc}, 1.0);

    dolfinx::common::Timer mtimer("% CSR Matvec");
    for (int i = 0; i < nreps; ++i)
    {
      std::fill(z.array().begin(), z.array().end(), 0);

      mat_op.mult(u, z);
    }
    mtimer.stop();

    T unorm = dolfinx::la::norm(u);
    T znorm = dolfinx::la::norm(z);
    // Compute error
    dolfinx::la::Vector<T> e(map, 1);

    auto axpy = [](auto&& r, auto alpha, auto&& x, auto&& y)
    {
      std::ranges::transform(x.array(), y.array(), r.array().begin(),
                             [alpha](auto x, auto y) { return alpha * x + y; });
    };

    axpy(e, T{-1}, y, z);
    T enorm = dolfinx::la::norm(e);

    b_results.znorm = znorm;
    b_results.enorm = enorm;

    if (rank == 0)
    {
      std::cout << "Norm of u = " << unorm << std::endl;
      std::cout << "Norm of z = " << znorm << std::endl;
      std::cout << "Norm of error = " << enorm << std::endl;
      std::cout << "Relative norm of error = " << enorm / znorm << std::endl;
    }
  }

  return b_results;
}

/// @cond protect from doxygen
template benchdolfinx::BenchmarkResults
benchdolfinx::laplace_action_cpu<double>(
    const dolfinx::fem::Form<double>&, const dolfinx::fem::Form<double>&,
    const dolfinx::fem::DirichletBC<double>&, int, int, double, int, bool,
    bool);

template benchdolfinx::BenchmarkResults benchdolfinx::laplace_action_cpu<float>(
    const dolfinx::fem::Form<float>&, const dolfinx::fem::Form<float>&,
    const dolfinx::fem::DirichletBC<float>&, int, int, float, int, bool, bool);
/// @endcond

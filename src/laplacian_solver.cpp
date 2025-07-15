// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "laplacian_solver.hpp"
#include "laplacian.hpp"
#include <basix/quadrature.h>
#include <dolfinx/la/MatrixCSR.h>

#if defined(USE_CUDA) || defined(USE_HIP)

#include "csr.hpp"
#include "forms.hpp"
#include "geometry_gpu.hpp"
#include "laplacian_gpu.hpp"
#include "mesh.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/sequence.h>

using namespace benchdolfinx;

//----------------------------------------------------------------------------
template <typename T>
void benchdolfinx::laplace_action(const dolfinx::fem::Form<T>& a,
                                  const dolfinx::fem::Form<T>& L,
                                  const dolfinx::fem::DirichletBC<T>& bc,
                                  int degree, int qmode, T kappa, int nreps,
                                  bool use_gauss)
{
  auto V = a.function_spaces()[0];

  // Define vectors
  using DeviceVector = benchdolfinx::Vector<T>;

  // Input vector
  auto map = V->dofmap()->index_map;
  spdlog::info("Create device vector u");

  DeviceVector u(map, 1);
  u.set(T{0.0});

  // Output vector
  spdlog::info("Create device vector y");
  DeviceVector y(map, 1);
  y.set(0);

  // Create matrix free operator
  spdlog::info("Create MatFreeLaplacian");
  dolfinx::common::Timer op_create_timer("% Create matfree operator");

  basix::quadrature::type quad_type
      = use_gauss ? basix::quadrature::type::gauss_jacobi
                  : basix::quadrature::type::gll;

  MatFreeLaplacian<T> op(*V, bc, degree, qmode, kappa, quad_type);

  op_create_timer.stop();

  dolfinx::la::Vector<T> b(map, 1);
  dolfinx::fem::assemble_vector(b.mutable_array(), L);
  u.copy_from_host(b); // Copy data from host vector to device vector
  u.scatter_fwd();

  // Matrix free
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nreps; ++i)
    op.apply(u, y);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;

  T unorm = benchdolfinx::norm(u);
  T ynorm = benchdolfinx::norm(y);

  int rank = dolfinx::MPI::rank(V->mesh()->comm());
  // Json::Value& out_root = root["results"];
  if (rank == 0)
  {
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    // std::cout << "Mat-free action Gdofs/s: "
    //           << ndofs_global * nreps / (1e9 * duration.count()) <<
    //           std::endl;

    // out_root["gdofs"] = ndofs_global * nreps / (1e9 * duration.count());

    std::cout << "Norm of u = " << unorm << std::endl;
    std::cout << "Norm of y = " << ynorm << std::endl;
  }

  bool matrix_comparison = true;
  if (matrix_comparison)
  {
    // Compare to assembling on CPU and copying matrix to GPU
    DeviceVector z(map, 1);
    z.set(T{0.0});

    benchdolfinx::MatrixOperator<T> mat_op(a, {bc});
    dolfinx::common::Timer mtimer("% CSR Matvec");
    for (int i = 0; i < nreps; ++i)
      mat_op.apply(u, z);
    mtimer.stop();

    T unorm = benchdolfinx::norm(u);
    T znorm = benchdolfinx::norm(z);
    // Compute error
    DeviceVector e(map, 1);
    benchdolfinx::axpy(e, T{-1}, y, z);
    T enorm = benchdolfinx::norm(e);

    if (rank == 0)
    {
      std::cout << "Norm of u = " << unorm << std::endl;
      std::cout << "Norm of z = " << znorm << std::endl;
      std::cout << "Norm of error = " << enorm << std::endl;
      std::cout << "Relative norm of error = " << enorm / znorm << std::endl;
      // out_root["error_norm"] = enorm;
    }
  }
}
//----------------------------------------------------------------------------
#else // CPU
template <typename T>
void benchdolfinx::laplace_action(const dolfinx::fem::Form<T>& a,
                                  const dolfinx::fem::Form<T>& L,
                                  const dolfinx::fem::DirichletBC<T>& bc,
                                  int degree, int qmode, T kappa, int nreps,
                                  bool use_gauss)
{
  auto V = a.function_spaces()[0];

  // Input vector
  auto map = V->dofmap()->index_map;

  spdlog::info("Create vector u");
  dolfinx::la::Vector<T> u(map, 1);
  u.set(0.0);

  // Output vector
  spdlog::info("Create vector y");
  dolfinx::la::Vector<T> y(map, 1);
  y.set(0.0);

  // Create matrix free operator
  spdlog::info("Create MatFreeLaplacian");
  dolfinx::common::Timer op_create_timer("% Create matfree operator");

  basix::quadrature::type quad_type
      = use_gauss ? basix::quadrature::type::gauss_jacobi
                  : basix::quadrature::type::gll;

  MatFreeLaplacian<T> op(*V, bc, degree, qmode, kappa, quad_type);

  op_create_timer.stop();

  dolfinx::fem::assemble_vector(u.mutable_array(), L);
  u.scatter_fwd();

  // Matrix free
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nreps; ++i)
    op.apply(u, y);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;

  T unorm = dolfinx::la::norm(u);
  T ynorm = dolfinx::la::norm(y);

  int rank = dolfinx::MPI::rank(V->mesh()->comm());
  // Json::Value& out_root = root["results"];
  if (rank == 0)
  {
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    // std::cout << "Mat-free action Gdofs/s: "
    //           << ndofs_global * nreps / (1e9 * duration.count()) <<
    //           std::endl;

    // out_root["gdofs"] = ndofs_global * nreps / (1e9 * duration.count());

    std::cout << "Norm of u = " << unorm << std::endl;
    std::cout << "Norm of y = " << ynorm << std::endl;
  }

  bool matrix_comparison = true;
  if (matrix_comparison)
  {
    dolfinx::la::Vector<T> z(map, 1);
    z.set(T{0.0});

    dolfinx::la::SparsityPattern sp = dolfinx::fem::create_sparsity_pattern(a);
    sp.finalize();
    dolfinx::la::MatrixCSR<T> mat_op(sp);
    dolfinx::fem::assemble_matrix(mat_op.mat_add_values(), a, {bc});
    dolfinx::common::Timer mtimer("% CSR Matvec");
    // for (int i = 0; i < nreps; ++i)
    //      mat_op.apply(u, z);
    mtimer.stop();

    T unorm = dolfinx::la::norm(u);
    T znorm = dolfinx::la::norm(z);
    // Compute error
    dolfinx::la::Vector<T> e(map, 1);

    auto axpy = [](auto&& r, auto alpha, auto&& x, auto&& y)
    {
      std::ranges::transform(x.array(), y.array(), r.mutable_array().begin(),
                             [alpha](auto x, auto y) { return alpha * x + y; });
    };

    axpy(e, T{-1}, y, z);
    T enorm = dolfinx::la::norm(e);

    if (rank == 0)
    {
      std::cout << "Norm of u = " << unorm << std::endl;
      std::cout << "Norm of z = " << znorm << std::endl;
      std::cout << "Norm of error = " << enorm << std::endl;
      std::cout << "Relative norm of error = " << enorm / znorm << std::endl;
      // out_root["error_norm"] = enorm;
    }
  }
}
#endif

/// @cond protect from doxygen
template void benchdolfinx::laplace_action<double>(
    const dolfinx::fem::Form<double>&, const dolfinx::fem::Form<double>&,
    const dolfinx::fem::DirichletBC<double>&, int, int, double, int, bool);
/// @endcond

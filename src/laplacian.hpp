// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "geometry_cpu.hpp"
#include "laplacian_cpu.hpp"
#include "mesh.hpp"
#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(USE_CUDA) || defined(USE_HIP) // GPU
#include "geometry_gpu.hpp"
#include "laplacian_gpu.hpp"
#include "vector.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

namespace benchdolfinx
{
/// @brief Check if a 2D array (matrix) is the identity matrix
/// @tparam T Type
/// @param[in] mat Matrix data in row-major order.
/// @param[in] shape Shape of the matrix.
/// @return True iff the matrix is an identity matrix, otherwise false.
template <typename T>
bool matrix_is_identity(const std::vector<T>& mat,
                        std::array<std::size_t, 2> shape)
{
  T eps = std::numeric_limits<T>::epsilon();
  if (shape[0] == shape[1])
  {
    for (std::size_t i = 0; i < shape[0]; ++i)
    {
      for (std::size_t j = 0; j < shape[1]; ++j)
      {
        if (i != j and std::abs(mat[i * shape[1] + j]) > 5 * eps)
          return false;
        else if (i == j and std::abs(mat[i * shape[1] + j] - 1.0) > 5 * eps)
          return false;
      }
    }
    return true;
  }
  else
    return false;
}

namespace impl
{
/// @brief Return the number of cells in the mesh
/// @param mesh Mesh
/// @tparam T Scalar type of mesh
/// @returns number of cells in the mesh
template <typename T>
std::size_t num_cells(const dolfinx::mesh::Mesh<T>& mesh)
{
  int tdim = mesh.topology()->dim();
  return mesh.topology()->index_map(tdim)->size_local()
         + mesh.topology()->index_map(tdim)->num_ghosts();
}

/// @brief Create markers for boundary conditions
/// @param bc Dirichlet boundary condition
/// @tparam T Scalar type
/// @returns List of markers for each DoF, zero if no BC, and one if a BC is
/// set.
template <typename T>
std::vector<std::int8_t>
build_bc_markers(const dolfinx::fem::DirichletBC<T>& bc)
{
  auto map = bc.function_space()->dofmap()->index_map;
  std::int32_t num_dofs = map->size_local() + map->num_ghosts();
  std::vector<std::int8_t> bc_marker(num_dofs, 0);
  bc.mark_dofs(bc_marker);
  return bc_marker;
}

/// @brief Basis function and quadrature tables for the matrix-free
/// operator, computed on the host.
template <typename T>
struct OperatorTables
{
  /// Number of quadrature points in 1D
  int nq;

  /// True if the interpolation from the element (degree P) to the 1D
  /// element with nodes at the quadrature points is the identity
  bool is_identity;

  /// 1D interpolation matrix, shape (nq, P + 1)
  std::vector<T> phi0;

  /// Derivatives of the 1D quadrature-point basis at the quadrature
  /// points, shape (nq, nq)
  std::vector<T> dphi1;

  /// Derivatives of the coordinate element basis at the 3D quadrature
  /// points, shape (3, nq^3, num coordinate dofs)
  std::vector<T> dphi_geometry;

  /// 3D quadrature weights, shape (nq^3)
  std::vector<T> weights;
};

/// @brief Compute the basis function and quadrature tables for the
/// matrix-free operator.
/// @param cmap Coordinate element of the mesh
/// @param degree Polynomial degree of operator
/// @param qmode Quadrature mode (0 or 1)
/// @param quad_type Quadrature type (GLL or Gauss)
template <typename T>
OperatorTables<T>
create_operator_tables(const dolfinx::fem::CoordinateElement<T>& cmap,
                       int degree, int qmode, basix::quadrature::type quad_type)
{
  basix::element::lagrange_variant variant;
  std::function<int(int)> q_map;
  if (quad_type == basix::quadrature::type::gauss_jacobi)
  {
    variant = basix::element::lagrange_variant::gl_warped;
    q_map = [](int p) { return 2 * p; };
  }
  else if (quad_type == basix::quadrature::type::gll)
  {
    variant = basix::element::lagrange_variant::gll_warped;
    q_map = [](int p) { return (p > 2) ? 2 * p - 2 : 2 * p - 1; };
  }
  else
  {
    throw std::runtime_error(
        "Unsupported quadrature type for mat-free operator");
  }

  OperatorTables<T> tables;

  // NOTE: Basix generates quadrature points in tensor-product
  // ordering, so this is OK
  auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
      quad_type, basix::cell::type::hexahedron, basix::polyset::type::standard,
      q_map(degree + qmode));

  std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(1, Gweights.size());
  std::vector<T> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

  // Keep dphi (skipping phi in table)
  tables.dphi_geometry.assign(phi_b.begin() + phi_b.size() / 4, phi_b.end());
  tables.weights = std::move(Gweights);

  // Create 1D element
  basix::FiniteElement<T> element0 = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, degree,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  // Create quadrature
  auto [points, weights] = basix::quadrature::make_quadrature<T>(
      quad_type, basix::cell::type::interval, basix::polyset::type::standard,
      q_map(degree + qmode));

  // Make sure geometry weights for 3D cell match size of 1D
  // quadrature weights
  tables.nq = weights.size();
  if (tables.weights.size()
      != static_cast<std::size_t>(tables.nq * tables.nq * tables.nq))
  {
    throw std::runtime_error("3D and 1D weight mismatch");
  }

  // Create higher-order 1D element for which the dofs coincide with
  // the quadrature points
  basix::FiniteElement<T> element1 = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, tables.nq - 1,
      variant, basix::element::dpc_variant::unset, true);

  // Compute interpolation matrix from element0 to element1
  auto [mat, shape_I]
      = basix::compute_interpolation_operator(element0, element1);

  // Check whether the interpolation matrix is the identity
  T precision = std::numeric_limits<T>::epsilon();
  for (auto& v : mat)
  {
    if (std::abs(v) < 5 * precision)
      v = 0;
  }
  tables.is_identity = matrix_is_identity(mat, shape_I);

  spdlog::info("Identity: {}", tables.is_identity);
  if (qmode == 0 and !tables.is_identity)
    throw std::runtime_error("Expecting identity matrix for qmode=0");

  tables.phi0 = std::move(mat);

  // Tabulate 1D and keep the derivative (second half of table)
  auto [table, shape] = element1.tabulate(1, points, {weights.size(), 1});
  tables.dphi1.assign(std::next(table.begin(), table.size() / 2), table.end());

  return tables;
}

/// @brief Call `f.template operator()<P, Q>()`, where the compile-time
/// polynomial degree `P` and number of 1D quadrature points `Q` are
/// equal to `degree` and `nq`.
///
/// Supports degree 1 to 7, with `nq = degree + 1` or `nq = degree + 2`.
template <int P = 1, typename F>
void dispatch_degree(int degree, int nq, F&& f)
{
  if constexpr (P > 7)
    throw std::runtime_error("Unsupported degree [operator]");
  else if (degree != P)
    dispatch_degree<P + 1>(degree, nq, std::forward<F>(f));
  else if (nq == P + 1)
    f.template operator()<P, P + 1>();
  else if (nq == P + 2)
    f.template operator()<P, P + 2>();
  else
    throw std::runtime_error("Unsupported nq");
}

/// @brief Call `f.template operator()<Q>()`, where the compile-time
/// number of 1D quadrature points `Q` is equal to `nq`.
///
/// Supports nq = 2 to 9.
template <int Q = 2, typename F>
void dispatch_nq(int nq, F&& f)
{
  if constexpr (Q > 9)
    throw std::runtime_error("Unsupported degree [geometry]: "
                             + std::to_string(nq));
  else if (nq != Q)
    dispatch_nq<Q + 1>(nq, std::forward<F>(f));
  else
    f.template operator()<Q>();
}
} // namespace impl

#if defined(USE_CUDA) || defined(USE_HIP)
/// GPU
template <typename T>
class MatFreeLaplacianGPU
{
public:
  using value_type = T;

  /// @brief Matrix-free Laplacian operator
  /// @param V FunctionSpace on which the operator is built
  /// @param bc Boundary condition, defining constrained degrees of freedom
  /// @param degree Polynomial degree of operator
  /// @param qmode Quadrature mode (0 or 1)
  /// @param constant Coefficient value, used on all cells
  /// @param quad_type Quadrature type (GLL or Gauss)
  MatFreeLaplacianGPU(const dolfinx::fem::FunctionSpace<T>& V,
                      const dolfinx::fem::DirichletBC<T>& bc, int degree,
                      int qmode, T constant, basix::quadrature::type quad_type)
      : _degree(degree), _cell_constants(impl::num_cells(*V.mesh()), constant),
        _cell_dofmap(V.dofmap()->map().data_handle(),
                     V.dofmap()->map().data_handle()
                         + V.dofmap()->map().size()),
        _bc_marker(impl::build_bc_markers(bc))
  {
    auto [lcells, bcells] = benchdolfinx::compute_boundary_cells(
        *V.mesh()->topology(), *V.dofmap());
    _lcells.assign(lcells.begin(), lcells.end());
    _bcells.assign(bcells.begin(), bcells.end());

    const dolfinx::mesh::Geometry<T>& geometry = V.mesh()->geometry();
    impl::OperatorTables<T> tables = impl::create_operator_tables<T>(
        geometry.cmaps().front(), _degree, qmode, quad_type);
    _op_nq = tables.nq;
    _is_identity = tables.is_identity;

    spdlog::debug("Copy interpolation matrix to device ({} bytes)",
                  tables.phi0.size() * sizeof(T));
    _phi0_const.assign(tables.phi0.begin(), tables.phi0.end());
    _dphi1_const.assign(tables.dphi1.begin(), tables.dphi1.end());

    // Precompute geometry. The input data is only needed here, so is
    // freed on return.
    spdlog::info("Precomputing geometry");
    std::span<const T> x = geometry.x();
    thrust::device_vector<T> xgeom(x.begin(), x.end());
    auto gdofmap = geometry.dofmap();
    thrust::device_vector<std::int32_t> geometry_dofmap(
        gdofmap.data_handle(), gdofmap.data_handle() + gdofmap.size());
    thrust::device_vector<T> dphi_geometry(tables.dphi_geometry.begin(),
                                           tables.dphi_geometry.end());
    thrust::device_vector<T> weights(tables.weights.begin(),
                                     tables.weights.end());

    // Local cells, then boundary cells (the order used in apply)
    thrust::device_vector<std::int32_t> cells(lcells.size() + bcells.size());
    thrust::copy(lcells.begin(), lcells.end(), cells.begin());
    thrust::copy(bcells.begin(), bcells.end(), cells.begin() + lcells.size());

    impl::dispatch_nq(
        _op_nq,
        [&]<int Q>()
        {
          _g_entity.resize(weights.size() * cells.size() * 6);
          dim3 block_size(weights.size());
          dim3 grid_size(cells.size());
          spdlog::info("Calling geometry_computation [Q={}, cells={}]", Q,
                       cells.size());
          geometry_computation_gpu<T, Q><<<grid_size, block_size>>>(
              thrust::raw_pointer_cast(xgeom.data()),
              thrust::raw_pointer_cast(_g_entity.data()),
              thrust::raw_pointer_cast(geometry_dofmap.data()),
              thrust::raw_pointer_cast(dphi_geometry.data()),
              thrust::raw_pointer_cast(weights.data()),
              thrust::raw_pointer_cast(cells.data()), cells.size());
          check_device_last_error();
        });
    device_synchronize();

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

  /// @brief Apply Laplacian operator
  /// @param in Input vector
  /// @param out Output vector
  template <typename Vector>
  void apply(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    set_value(out, T{0});
    impl::dispatch_degree(_degree, _op_nq, [&]<int P, int Q>()
                          { this->template impl_operator<P, Q>(in, out); });
    spdlog::debug("Mat free operator end");
  }

private:
  /// @brief Implementation of the action of the operator
  /// @tparam Vector Vector Type
  /// @tparam P Polynomial degree of the operator
  /// @tparam Q Number of quadrature points (1D)
  /// @param in Input vector
  /// @param out Output vector, with values representing the laplacian of the
  /// input vector
  template <int P, int Q, typename Vector>
  void impl_operator(Vector& in, Vector& out)
  {
    in.scatter_fwd_begin(get_pack_fn<T>(512),
                         [](auto&& x) { return x.data().get(); });

    T* geometry_ptr = thrust::raw_pointer_cast(_g_entity.data());

    // Cells which only use owned dofs, computed while the ghost update
    // is in progress
    if (!_lcells.empty())
    {
      spdlog::debug("Calling stiffness_operator on local cells [{}]",
                    _lcells.size());
      T* x = in.array().data().get();
      T* y = out.array().data().get();

      dim3 block_size(Q, Q, Q);
      dim3 grid_size(_lcells.size());
      stiffness_operator_gpu<T, P, Q><<<grid_size, block_size>>>(
          x, thrust::raw_pointer_cast(_cell_constants.data()), y, geometry_ptr,
          thrust::raw_pointer_cast(_phi0_const.data()),
          thrust::raw_pointer_cast(_dphi1_const.data()),
          thrust::raw_pointer_cast(_cell_dofmap.data()),
          thrust::raw_pointer_cast(_lcells.data()), _lcells.size(),
          thrust::raw_pointer_cast(_bc_marker.data()), _is_identity);

      check_device_last_error();
    }

    in.scatter_fwd_end(get_unpack_insert_fn<T>(512));

    if (!_bcells.empty())
    {
      spdlog::debug("Calling stiffness_operator on boundary cells [{}]",
                    _bcells.size());

      geometry_ptr += 6 * Q * Q * Q * _lcells.size();

      T* x = in.array().data().get();
      T* y = out.array().data().get();

      dim3 block_size(Q, Q, Q);
      dim3 grid_size(_bcells.size());
      stiffness_operator_gpu<T, P, Q><<<grid_size, block_size>>>(
          x, thrust::raw_pointer_cast(_cell_constants.data()), y, geometry_ptr,
          thrust::raw_pointer_cast(_phi0_const.data()),
          thrust::raw_pointer_cast(_dphi1_const.data()),
          thrust::raw_pointer_cast(_cell_dofmap.data()),
          thrust::raw_pointer_cast(_bcells.data()), _bcells.size(),
          thrust::raw_pointer_cast(_bc_marker.data()), _is_identity);

      check_device_last_error();
    }

    device_synchronize();
  }

  // Polynomial degree
  int _degree;

  // Number of quadrature points in 1D
  int _op_nq;

  // On-device storage for constants (by cell index) and dofmap
  thrust::device_vector<T> _cell_constants;
  thrust::device_vector<std::int32_t> _cell_dofmap;

  thrust::device_vector<std::int8_t> _bc_marker;

  // Geometry data at quadrature points, for local cells then boundary
  // cells
  thrust::device_vector<T> _g_entity;

  // Interpolation is the identity
  bool _is_identity;

  // Exclusively owned cells (do not share dofs with other processes)
  thrust::device_vector<int> _lcells;

  // Cells on partition boundaries
  thrust::device_vector<int> _bcells;

  /// phi0 Input basis function table
  thrust::device_vector<T> _phi0_const;

  /// dphi1 Input basis function derivative table
  thrust::device_vector<T> _dphi1_const;
};
#endif

template <typename T>
class MatFreeLaplacianCPU
{
public:
  using value_type = T;

  /// @brief Matrix-free Laplacian operator
  /// @param V FunctionSpace on which the operator is built
  /// @param bc Boundary condition, defining constrained degrees of freedom
  /// @param degree Polynomial degree of operator
  /// @param qmode Quadrature mode (only 0 is supported)
  /// @param constant Coefficient value, used on all cells
  /// @param quad_type Quadrature type (GLL or Gauss)
  MatFreeLaplacianCPU(const dolfinx::fem::FunctionSpace<T>& V,
                      const dolfinx::fem::DirichletBC<T>& bc, int degree,
                      int qmode, T constant, basix::quadrature::type quad_type)
      : _degree(degree), _cell_constants(impl::num_cells(*V.mesh()), constant),
        _cell_dofmap(V.dofmap()->map().data_handle(),
                     V.dofmap()->map().data_handle()
                         + V.dofmap()->map().size()),
        _bc_marker(impl::build_bc_markers(bc))
  {
    if (qmode != 0)
      throw std::runtime_error("MatFreeLaplacianCPU supports qmode=0 only");

    auto [lcells, bcells] = benchdolfinx::compute_boundary_cells(
        *V.mesh()->topology(), *V.dofmap());
    _lcells = std::move(lcells);
    _bcells = std::move(bcells);

    // For qmode=0 the interpolation to the quadrature-point basis is
    // the identity (checked in create_operator_tables), so phi0 is not
    // needed
    const dolfinx::mesh::Geometry<T>& geometry = V.mesh()->geometry();
    impl::OperatorTables<T> tables = impl::create_operator_tables<T>(
        geometry.cmaps().front(), _degree, qmode, quad_type);
    _op_nq = tables.nq;
    _dphi1_const = std::move(tables.dphi1);

    // Precompute geometry for local cells, then boundary cells (the
    // order used in apply)
    spdlog::info("Precomputing geometry");
    std::vector<std::int32_t> cells(_lcells.size() + _bcells.size());
    std::copy(_lcells.begin(), _lcells.end(), cells.begin());
    std::copy(_bcells.begin(), _bcells.end(), cells.begin() + _lcells.size());

    auto gdofmap = geometry.dofmap();
    impl::dispatch_nq(
        _op_nq,
        [&]<int Q>()
        {
          _g_entity.resize(tables.weights.size() * cells.size() * 6);
          spdlog::info("Calling geometry_computation [Q={}, cells={}]", Q,
                       cells.size());
          geometry_computation_cpu<T, Q>(
              geometry.x(), _g_entity,
              std::span<const std::int32_t>(gdofmap.data_handle(),
                                            gdofmap.size()),
              tables.dphi_geometry, tables.weights, cells);
        });

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

  /// @brief Apply Laplacian operator
  /// @param in Input vector
  /// @param out Output vector
  template <typename Vector>
  void apply(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    std::fill(out.array().begin(), out.array().end(), 0);
    impl::dispatch_degree(_degree, _op_nq, [&]<int P, int Q>()
                          { this->template impl_operator<P, Q>(in, out); });
    spdlog::debug("Mat free operator end");
  }

private:
  /// @brief Implementation of the action of the operator
  /// @tparam Vector Vector Type
  /// @tparam P Polynomial degree of the operator
  /// @tparam Q Number of quadrature points (1D)
  /// @param in Input vector
  /// @param out Output vector, with values representing the laplacian of the
  /// input vector
  template <int P, int Q, typename Vector>
  void impl_operator(Vector& in, Vector& out)
  {
    // The CPU kernel supports qmode=0 (Q = P + 1) only
    if constexpr (Q != P + 1)
      throw std::runtime_error("MatFreeLaplacianCPU supports qmode=0 only");
    else
    {
      in.scatter_fwd_begin();

      T* geometry_ptr = _g_entity.data();

      // Cells which only use owned dofs, computed while the ghost
      // update is in progress
      if (!_lcells.empty())
      {
        spdlog::debug("Calling stiffness_operator on local cells [{}]",
                      _lcells.size());
        stiffness_operator_cpu<T, P, Q>(
            in.array().data(), _cell_constants.data(), out.array().data(),
            geometry_ptr, _dphi1_const.data(), _cell_dofmap.data(),
            _lcells.data(), _lcells.size(), _bc_marker.data());
      }

      in.scatter_fwd_end();

      if (!_bcells.empty())
      {
        spdlog::debug("Calling stiffness_operator on boundary cells [{}]",
                      _bcells.size());
        geometry_ptr += 6 * Q * Q * Q * _lcells.size();
        stiffness_operator_cpu<T, P, Q>(
            in.array().data(), _cell_constants.data(), out.array().data(),
            geometry_ptr, _dphi1_const.data(), _cell_dofmap.data(),
            _bcells.data(), _bcells.size(), _bc_marker.data());
      }
    }
  }

  // Polynomial degree
  int _degree;

  // Number of quadrature points in 1D
  int _op_nq;

  // Constants (by cell index) and dofmap
  std::vector<T> _cell_constants;
  std::vector<std::int32_t> _cell_dofmap;

  std::vector<std::int8_t> _bc_marker;

  // Geometry data at quadrature points, for local cells then boundary
  // cells
  std::vector<T> _g_entity;

  // Exclusively owned cells (do not share dofs with other processes)
  std::vector<std::int32_t> _lcells;

  // Cells on partition boundaries
  std::vector<std::int32_t> _bcells;

  /// dphi1 Input basis function derivative table
  std::vector<T> _dphi1_const;
};
} // namespace benchdolfinx

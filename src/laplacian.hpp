// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "mesh.hpp"
#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>

#if defined(USE_CUDA) || defined(USE_HIP)
#include "geometry_gpu.hpp"
#include "laplacian_gpu.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#else
#include "geometry_cpu.hpp"
#include "laplacian_cpu.hpp"
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
} // namespace impl

#if defined(USE_CUDA) || defined(USE_HIP)
template <typename T>
class MatFreeLaplacian
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
  MatFreeLaplacian(const dolfinx::fem::FunctionSpace<T>& V,
                   const dolfinx::fem::DirichletBC<T>& bc, int degree,
                   int qmode, T constant, basix::quadrature::type quad_type)
      : _degree(degree), _cell_constants(impl::num_cells(*V.mesh()), constant),
        _cell_dofmap(V.dofmap()->map().data_handle(),
                     V.dofmap()->map().data_handle()
                         + V.dofmap()->map().size()),
        _xgeom(V.mesh()->geometry().x().begin(),
               V.mesh()->geometry().x().end()),
        _geometry_dofmap(V.mesh()->geometry().dofmap().data_handle(),
                         V.mesh()->geometry().dofmap().data_handle()
                             + V.mesh()->geometry().dofmap().size()),
        _bc_marker(impl::build_bc_markers(bc))
  {
    {
      auto [lcells, bcells] = benchdolfinx::compute_boundary_cells<T>(V);
      _lcells = lcells;
      _bcells = bcells;
    }

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

    // NOTE: Basix generates quadrature points in tensor-product
    // ordering, so this is OK

    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::hexahedron,
        basix::polyset::type::standard, q_map(_degree + qmode));

    const dolfinx::fem::CoordinateElement<T>& cmap
        = V.mesh()->geometry().cmap();
    std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(1, Gweights.size());
    std::vector<T> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

    // Copy dphi to device (skipping phi in table)
    _dphi_geometry.assign(phi_b.begin() + phi_b.size() / 4, phi_b.end());
    _g_weights.assign(Gweights.begin(), Gweights.end());

    // Create 1D element
    basix::FiniteElement<T> element0 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, _degree,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::interval, basix::polyset::type::standard,
        q_map(_degree + qmode));

    // Make sure geometry weights for 3D cell match size of 1D
    // quadrature weights
    _op_nq = weights.size();
    if (Gweights.size() != _op_nq * _op_nq * _op_nq)
      throw std::runtime_error("3D and 1D weight mismatch");

    // Create higher-order 1D element for which the dofs coincide with
    // the quadrature points
    basix::FiniteElement<T> element1 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, _op_nq - 1,
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
    _is_identity = matrix_is_identity(mat, shape_I);

    spdlog::info("Identity: {}", _is_identity);
    if (qmode == 0 and !_is_identity)
      throw std::runtime_error("Expecting identity matrix for qmode=0");

    // Tabulate 1D
    auto [table, shape] = element1.tabulate(1, points, {weights.size(), 1});

    // Copy interpolation matrix to device
    spdlog::debug("Copy interpolation matrix to device ({} bytes)",
                  mat.size() * sizeof(T));
    _phi0_const.resize(mat.size());
    thrust::copy(mat.begin(), mat.end(), _phi0_const.begin());

    // Copy derivative to device (second half of table)
    _dphi1_const.resize(table.size() / 2);
    thrust::copy(std::next(table.begin(), table.size() / 2), table.end(),
                 _dphi1_const.begin());

    spdlog::info("Precomputing geometry");
    thrust::device_vector<std::int32_t> cells_d(_lcells.size()
                                                + _bcells.size());
    thrust::copy(_lcells.begin(), _lcells.end(), cells_d.begin());
    thrust::copy(_bcells.begin(), _bcells.end(),
                 cells_d.begin() + _lcells.size());
    std::span<const std::int32_t> cell_list_d(
        thrust::raw_pointer_cast(cells_d.data()), cells_d.size());

    compute_geometry(_op_nq, cell_list_d);
    device_synchronize();

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

private:
  /// @brief Precompute weighted geometry data on GPU
  /// Computes the symmetric 3x3 geometric tensor,
  /// and stores as 6 values for each quadrature point of each cell.
  /// @note The quadrature weights are also combined into the values of the
  /// geometric tensor at each point.
  /// @param nq Number of quadrature points in 1D (total points per cell will
  /// be nq^3)
  /// @param cell_list List of cell indices to compute for
  template <int Q = 2>
  void compute_geometry(int nq, std::span<const int> cell_list)
  {
    if constexpr (Q < 10)
    {
      if (nq > Q)
        compute_geometry<Q + 1>(nq, cell_list);
      else
      {
        assert(nq == Q);
        _g_entity.resize(_g_weights.size() * cell_list.size() * 6);
        dim3 block_size(_g_weights.size());
        dim3 grid_size(cell_list.size());

        spdlog::info("xgeom size {}", _xgeom.size());
        spdlog::info("G_entity size {}", _g_entity.size());
        spdlog::info("geometry_dofmap size {}", _geometry_dofmap.size());
        spdlog::info("dphi_geometry size {}", _dphi_geometry.size());
        spdlog::info("Gweights size {}", _g_weights.size());
        spdlog::info("cell_list size {}", cell_list.size());
        spdlog::info("Calling geometry_computation [{} {}]", Q, nq);

        std::size_t shm_size = 24 * sizeof(T); // coordinate size (8x3)
        geometry_computation<T, Q><<<grid_size, block_size, shm_size, 0>>>(
            thrust::raw_pointer_cast(_xgeom.data()),
            thrust::raw_pointer_cast(_g_entity.data()),
            thrust::raw_pointer_cast(_geometry_dofmap.data()),
            thrust::raw_pointer_cast(_dphi_geometry.data()),
            thrust::raw_pointer_cast(_g_weights.data()), cell_list.data(),
            cell_list.size());
        spdlog::debug("Done geometry_computation");
      }
    }
    else
      throw std::runtime_error("Unsupported degree [geometry]");
  }

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
    spdlog::debug("impl_operator operator start");

    in.scatter_fwd_begin();

    T* geometry_ptr = thrust::raw_pointer_cast(_g_entity.data());

    if (!_lcells.empty())
    {
      spdlog::debug("Calling stiffness_operator on local cells [{}]",
                    _lcells.size());
      T* x = in.mutable_array().data();
      T* y = out.mutable_array().data();

      dim3 block_size(Q, Q, Q);
      dim3 grid_size(_lcells.size());
      stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
          x, thrust::raw_pointer_cast(_cell_constants.data()), y, geometry_ptr,
          thrust::raw_pointer_cast(_phi0_const.data()),
          thrust::raw_pointer_cast(_dphi1_const.data()),
          thrust::raw_pointer_cast(_cell_dofmap.data()),
          thrust::raw_pointer_cast(_lcells.data()), _lcells.size(),
          thrust::raw_pointer_cast(_bc_marker.data()), _is_identity);

      check_device_last_error();
    }

    spdlog::debug("impl_operator done lcells");

    spdlog::debug("cell_constants size {}", _cell_constants.size());
    spdlog::debug("in size {}", in.array().size());
    spdlog::debug("out size {}", out.array().size());
    spdlog::debug("G_entity size {}", _g_entity.size());
    spdlog::debug("cell_dofmap size {}", _cell_dofmap.size());
    spdlog::debug("bc_marker size {}", _bc_marker.size());

    in.scatter_fwd_end();

    spdlog::debug("impl_operator after scatter");

    if (!_bcells.empty())
    {
      spdlog::debug("impl_operator doing bcells. bcells size = {}",
                    _bcells.size());

      geometry_ptr += 6 * Q * Q * Q * _lcells.size();

      T* x = in.mutable_array().data();
      T* y = out.mutable_array().data();

      dim3 block_size(Q, Q, Q);
      dim3 grid_size(_bcells.size());
      stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
          x, thrust::raw_pointer_cast(_cell_constants.data()), y, geometry_ptr,
          thrust::raw_pointer_cast(_phi0_const.data()),
          thrust::raw_pointer_cast(_dphi1_const.data()),
          thrust::raw_pointer_cast(_cell_dofmap.data()),
          thrust::raw_pointer_cast(_bcells.data()), _bcells.size(),
          thrust::raw_pointer_cast(_bc_marker.data()), _is_identity);

      check_device_last_error();
    }

    device_synchronize();
    spdlog::debug("impl_operator done bcells");
  }

public:
  /// @brief Apply Laplacian operator
  /// @param in Input vector
  /// @param out Output vector
  template <typename Vector>
  void apply(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    out.set(0);

    if (_op_nq == _degree + 1)
    {
      if (_degree == 1)
        impl_operator<1, 2>(in, out);
      else if (_degree == 2)
        impl_operator<2, 3>(in, out);
      else if (_degree == 3)
        impl_operator<3, 4>(in, out);
      else if (_degree == 4)
        impl_operator<4, 5>(in, out);
      else if (_degree == 5)
        impl_operator<5, 6>(in, out);
      else if (_degree == 6)
        impl_operator<6, 7>(in, out);
      else if (_degree == 7)
        impl_operator<7, 8>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else if (_op_nq == _degree + 2)
    {
      if (_degree == 1)
        impl_operator<1, 3>(in, out);
      else if (_degree == 2)
        impl_operator<2, 4>(in, out);
      else if (_degree == 3)
        impl_operator<3, 5>(in, out);
      else if (_degree == 4)
        impl_operator<4, 6>(in, out);
      else if (_degree == 5)
        impl_operator<5, 7>(in, out);
      else if (_degree == 6)
        impl_operator<6, 8>(in, out);
      else if (_degree == 7)
        impl_operator<7, 9>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else
      throw std::runtime_error("Unsupported nq");

    spdlog::debug("Mat free operator end");
  }

private:
  // Polynomial degree
  std::size_t _degree;

  // Number of quadrature points in 1D
  std::size_t _op_nq;

  // Reference to on-device storage for constants, dofmap etc.
  thrust::device_vector<T> _cell_constants;
  thrust::device_vector<std::int32_t> _cell_dofmap;

  // Reference to on-device storage of geometry data
  thrust::device_vector<T> _xgeom;
  thrust::device_vector<std::int32_t> _geometry_dofmap;

  // geometry tables dphi on device
  thrust::device_vector<T> _dphi_geometry;

  thrust::device_vector<std::int8_t> _bc_marker;

  // On device storage for geometry quadrature weights
  thrust::device_vector<T> _g_weights;

  // On device storage for geometry data (computed for each batch of cells)
  thrust::device_vector<T> _g_entity;

  // Interpolation is the identity
  bool _is_identity;

  // Lists of cells which are local (lcells) and boundary (bcells)

  // Exclusively owned cells (not not share dofs with other processes)
  thrust::device_vector<int> _lcells;

  // Cells on partition boundaries
  thrust::device_vector<int> _bcells;

  /// phi0 Input basis function table
  thrust::device_vector<T> _phi0_const;

  /// dphi1 Input basis function derivative table
  thrust::device_vector<T> _dphi1_const;
};

#else

// CPU Version

template <typename T>
class MatFreeLaplacian
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
  MatFreeLaplacian(const dolfinx::fem::FunctionSpace<T>& V,
                   const dolfinx::fem::DirichletBC<T>& bc, int degree,
                   int qmode, T constant, basix::quadrature::type quad_type)
      : _degree(degree), _cell_constants(impl::num_cells(*V.mesh()), constant),
        _cell_dofmap(V.dofmap()->map().data_handle(),
                     V.dofmap()->map().data_handle()
                         + V.dofmap()->map().size()),
        _xgeom(V.mesh()->geometry().x().begin(),
               V.mesh()->geometry().x().end()),
        _geometry_dofmap(V.mesh()->geometry().dofmap().data_handle(),
                         V.mesh()->geometry().dofmap().data_handle()
                             + V.mesh()->geometry().dofmap().size()),
        _bc_marker(impl::build_bc_markers(bc))
  {
    {
      auto [lcells, bcells] = benchdolfinx::compute_boundary_cells<T>(V);
      _lcells = lcells;
      _bcells = bcells;
    }

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

    // NOTE: Basix generates quadrature points in tensor-product
    // ordering, so this is OK

    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::hexahedron,
        basix::polyset::type::standard, q_map(_degree + qmode));

    const dolfinx::fem::CoordinateElement<T>& cmap
        = V.mesh()->geometry().cmap();
    std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(1, Gweights.size());
    std::vector<T> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

    // Copy dphi to device (skipping phi in table)
    _dphi_geometry.assign(phi_b.begin() + phi_b.size() / 4, phi_b.end());
    _g_weights.assign(Gweights.begin(), Gweights.end());

    // Create 1D element
    basix::FiniteElement<T> element0 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, _degree,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::interval, basix::polyset::type::standard,
        q_map(_degree + qmode));

    // Make sure geometry weights for 3D cell match size of 1D
    // quadrature weights
    _op_nq = weights.size();
    if (Gweights.size() != _op_nq * _op_nq * _op_nq)
      throw std::runtime_error("3D and 1D weight mismatch");

    // Create higher-order 1D element for which the dofs coincide with
    // the quadrature points
    basix::FiniteElement<T> element1 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, _op_nq - 1,
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
    _is_identity = matrix_is_identity(mat, shape_I);

    spdlog::info("Identity: {}", _is_identity);
    if (qmode == 0 and !_is_identity)
      throw std::runtime_error("Expecting identity matrix for qmode=0");

    // Tabulate 1D
    auto [table, shape] = element1.tabulate(1, points, {weights.size(), 1});

    // Copy interpolation matrix to device
    spdlog::debug("Copy interpolation matrix to device ({} bytes)",
                  mat.size() * sizeof(T));
    _phi0_const.assign(mat.begin(), mat.end());

    // Copy derivative to device (second half of table)
    _dphi1_const.assign(std::next(table.begin(), table.size() / 2),
                        table.end());

    spdlog::info("Precomputing geometry");
    std::vector<std::int32_t> cells_d(_lcells.size() + _bcells.size());
    std::copy(_lcells.begin(), _lcells.end(), cells_d.begin());
    std::copy(_bcells.begin(), _bcells.end(), cells_d.begin() + _lcells.size());

    compute_geometry(_op_nq, cells_d);

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

  template <int Q = 2>
  void compute_geometry(int nq, std::span<const int> cell_list)
  {
    if constexpr (Q < 10)
    {
      if (nq > Q)
        compute_geometry<Q + 1>(nq, cell_list);
      else
      {
        assert(nq == Q);
        _g_entity.resize(_g_weights.size() * cell_list.size() * 6);

        spdlog::info("xgeom size {}", _xgeom.size());
        spdlog::info("G_entity size {}", _g_entity.size());
        spdlog::info("geometry_dofmap size {}", _geometry_dofmap.size());
        spdlog::info("dphi_geometry size {}", _dphi_geometry.size());
        spdlog::info("Gweights size {}", _g_weights.size());
        spdlog::info("cell_list size {}", cell_list.size());
        spdlog::info("Calling geometry_computation [{} {}]", Q, nq);

        geometry_computation<T, Q>(_xgeom.data(), _g_entity.data(),
                                   _geometry_dofmap.data(),
                                   _dphi_geometry.data(), _g_weights.data(),
                                   cell_list.data(), cell_list.size());

        spdlog::debug("Done geometry_computation");
      }
    }
    else
      throw std::runtime_error("Unsupported degree [geometry]: "
                               + std::to_string(nq));
  }

private:
  // Polynomial degree
  std::size_t _degree;

  // Number of quadrature points in 1D
  std::size_t _op_nq;

  // Reference to on-device storage for constants, dofmap etc.
  std::vector<T> _cell_constants;
  std::vector<std::int32_t> _cell_dofmap;

  // Reference to on-device storage of geometry data
  std::vector<T> _xgeom;
  std::vector<std::int32_t> _geometry_dofmap;

  // geometry tables dphi on device
  std::vector<T> _dphi_geometry;

  std::vector<std::int8_t> _bc_marker;

  // On device storage for geometry quadrature weights
  std::vector<T> _g_weights;

  // On device storage for geometry data (computed for each batch of cells)
  std::vector<T> _g_entity;

  // Interpolation is the identity
  bool _is_identity;

  // Lists of cells which are local (lcells) and boundary (bcells)

  // Exclusively owned cells (not not share dofs with other processes)
  std::vector<int> _lcells;

  // Cells on partition boundaries
  std::vector<int> _bcells;

  /// phi0 Input basis function table
  std::vector<T> _phi0_const;

  /// dphi1 Input basis function derivative table
  std::vector<T> _dphi1_const;
};
#endif
} // namespace benchdolfinx

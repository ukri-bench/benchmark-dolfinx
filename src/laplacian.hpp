// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#if defined(USE_CUDA) || defined(USE_HIP)

#include "geometry_gpu.hpp"
#include "laplacian_gpu.hpp"

#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace benchdolfinx
{
// template <typename T, int P, int Q>
// __constant__ T phi0_const[Q * (P + 1)];

// template <typename T, int Q>
// __constant__ T dphi1_const[Q * Q];

// /// @brief Compute 3d index from 1d indices.
// ///
// /// Compute the index `idx = ld0 * i + ld1 * j + ld2 * k`.
// ///
// /// For contiguous, row-major storage of a tensor with shape `(n0, n1,
// /// n2)`, use `ld0=n1*n2`, `ld1=n2`, `ld2=1` (`k` varies fastest,
// /// followed by `j`).
// ///
// /// For contiguous, column-major storage of a tensor with shape `(n0,
// /// n1, n2)`, use `ld0=1`, `ld1=n0`, `ld2=n0*n1` (`i` varies fastest,
// /// followed by `j`).
// ///
// /// For contiguous storage with `j` varying fastest and `i` slowest, use
// /// `ld0=n1*n2`, `ld1=1`, `ld2=n1`
// ///
// /// For contiguous storage with `j` varying fastest and `k` slowest, use
// /// `ld0=n1`, `ld1=1`, `ld2=n0*n1`
// ///
// /// @tparam ld0 Stride for first (`i`) index.
// /// @tparam ld1 Stride for second (`k`) index.
// /// @tparam ld2 Stride for third (`k`) index.
// /// @param[in] i
// /// @param[in] j
// /// @param[in] k
// /// @return Flattened index.
// template <int ld0, int ld1, int ld2>
// __device__ __forceinline__ int ijk(int i, int j, int k)
// {
//   return i * ld0 + j * ld1 + k * ld2;
// }

/// @brief TODO
/// @tparam T
/// @param mat
/// @param shape
/// @return
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

// FIXME Could just replace these maps with expression
static const std::map<int, int> q_map_gll
    = {{1, 1}, {2, 3}, {3, 4}, {4, 6}, {5, 8}, {6, 10}, {7, 12}, {8, 14}};

static const std::map<int, int> q_map_gq
    = {{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}, {6, 12}, {7, 14}, {8, 16}};

template <typename T>
class MatFreeLaplacian
{
public:
  using value_type = T;

  MatFreeLaplacian(int degree, int qmode, std::span<const T> coefficients,
                   std::span<const std::int32_t> dofmap,
                   std::span<const T> xgeom,
                   std::span<const std::int32_t> geometry_dofmap,
                   const dolfinx::fem::CoordinateElement<T>& cmap,
                   const std::vector<int>& lcells,
                   const std::vector<int>& bcells,
                   std::span<const std::int8_t> bc_marker,
                   basix::quadrature::type quad_type,
                   std::size_t batch_size = 0)
      : degree(degree), cell_constants(coefficients), cell_dofmap(dofmap),
        xgeom(xgeom), geometry_dofmap(geometry_dofmap), bc_marker(bc_marker),
        batch_size(batch_size)
  {
    basix::element::lagrange_variant variant;
    std::map<int, int> q_map;
    if (quad_type == basix::quadrature::type::gauss_jacobi)
    {
      variant = basix::element::lagrange_variant::gl_warped;
      q_map = q_map_gq;
    }
    else if (quad_type == basix::quadrature::type::gll)
    {
      variant = basix::element::lagrange_variant::gll_warped;
      q_map = q_map_gll;
    }
    else
      throw std::runtime_error(
          "Unsupported quadrature type for mat-free operator");

    // NOTE: Basix generates quadrature points in tensor-product ordering, so
    // this is OK
    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::hexahedron,
        basix::polyset::type::standard, q_map.at(degree + qmode));

    std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(1, Gweights.size());
    std::vector<T> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

    // Copy dphi to device (skipping phi in table)
    dphi_geometry.resize(phi_b.size() * 3 / 4);
    thrust::copy(phi_b.begin() + phi_b.size() / 4, phi_b.end(),
                 dphi_geometry.begin());

    Gweights_d.resize(Gweights.size());
    thrust::copy(Gweights.begin(), Gweights.end(), Gweights_d.begin());

    // Create 1D element
    basix::FiniteElement<T> element0 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, degree,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::interval, basix::polyset::type::standard,
        q_map.at(degree + qmode));

    // Make sure geometry weights for 3D cell match size of 1D
    // quadrature weights
    op_nq = weights.size();
    if (Gweights.size() != op_nq * op_nq * op_nq)
      throw std::runtime_error("3D and 1D weight mismatch");

    // Create higher-order 1D element for which the dofs coincide with
    // the quadrature points
    basix::FiniteElement<T> element1 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, op_nq - 1,
        variant, basix::element::dpc_variant::unset, true);

    // Compute interpolation matrix from element0 to element1
    auto [mat, shape_I]
        = basix::compute_interpolation_operator(element0, element1);

    T precision = std::numeric_limits<T>::epsilon();
    for (auto& v : mat)
    {
      if (std::abs(v) < 5 * precision)
        v = 0;
    }

    // Check whether the interpolation matrix is the identity
    is_identity = matrix_is_identity(mat, shape_I);

    spdlog::info("Identity: {}", is_identity);

    // Tabulate 1D
    auto [table, shape] = element1.tabulate(1, points, {weights.size(), 1});

    // Basis value gradient evualation table
    if (op_nq == degree + 1)
    {
      switch (degree)
      {
      case 1:
        copy_phi_tables<1, 2>(mat, table);
        break;
      case 2:
        copy_phi_tables<2, 3>(mat, table);
        break;
      case 3:
        copy_phi_tables<3, 4>(mat, table);
        break;
      case 4:
        copy_phi_tables<4, 5>(mat, table);
        break;
      case 5:
        copy_phi_tables<5, 6>(mat, table);
        break;
      case 6:
        copy_phi_tables<6, 7>(mat, table);
        break;
      case 7:
        copy_phi_tables<7, 8>(mat, table);
        break;
      default:
        throw std::runtime_error("Unsupported degree");
      }
    }
    else if (op_nq == degree + 2)
    {
      switch (degree)
      {
      case 1:
        copy_phi_tables<1, 3>(mat, table);
        break;
      case 2:
        copy_phi_tables<2, 4>(mat, table);
        break;
      case 3:
        copy_phi_tables<3, 5>(mat, table);
        break;
      case 4:
        copy_phi_tables<4, 6>(mat, table);
        break;
      case 5:
        copy_phi_tables<5, 7>(mat, table);
        break;
      case 6:
        copy_phi_tables<6, 8>(mat, table);
        break;
      case 7:
        copy_phi_tables<7, 9>(mat, table);
        break;
      default:
        throw std::runtime_error("Unsupported degree");
      }
    }
    else
      throw std::runtime_error("Unsupported nq");

    // Copy interpolation matrix to device
    spdlog::debug("Copy interpolation matrix to device ({} bytes)",
                  mat.size() * sizeof(T));

    // Copy lists of local and boundary cells to device
    lcells_device.resize(lcells.size());
    thrust::copy(lcells.begin(), lcells.end(), lcells_device.begin());
    bcells_device.resize(bcells.size());
    thrust::copy(bcells.begin(), bcells.end(), bcells_device.begin());

    // If we're not batching the geometry, precompute it
    if (batch_size == 0)
    {
      // FIXME Store cells and local/ghost offsets instead to avoid this?
      spdlog::info("Precomputing geometry");
      thrust::device_vector<std::int32_t> cells_d(lcells_device.size()
                                                  + bcells_device.size());
      thrust::copy(lcells_device.begin(), lcells_device.end(), cells_d.begin());
      thrust::copy(bcells_device.begin(), bcells_device.end(),
                   cells_d.begin() + lcells_device.size());
      std::span<std::int32_t> cell_list_d(
          thrust::raw_pointer_cast(cells_d.data()), cells_d.size());

      compute_geometry(op_nq, cell_list_d);
      device_synchronize();
    }

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

  /// @brief Compute weighted geometry data on GPU
  /// @param nq Number of quadrature points in 1D
  /// @param cell_list_d List of cell indices to compute for
  template <int Q = 2>
  void compute_geometry(int nq, std::span<int> cell_list_d)
  {
    if constexpr (Q < 10)
    {
      if (nq > Q)
        compute_geometry<Q + 1>(nq, cell_list_d);
      else
      {
        assert(nq == Q);
        G_entity.resize(Gweights_d.size() * cell_list_d.size() * 6);
        dim3 block_size(Gweights_d.size());
        dim3 grid_size(cell_list_d.size());

        spdlog::info("xgeom size {}", xgeom.size());
        spdlog::info("G_entity size {}", G_entity.size());
        spdlog::info("geometry_dofmap size {}", geometry_dofmap.size());
        spdlog::info("dphi_geometry size {}", dphi_geometry.size());
        spdlog::info("Gweights size {}", Gweights_d.size());
        spdlog::info("cell_list_d size {}", cell_list_d.size());
        spdlog::info("Calling geometry_computation [{} {}]", Q, nq);

        std::size_t shm_size = 24 * sizeof(T); // coordinate size (8x3)
        geometry_computation<T, Q><<<grid_size, block_size, shm_size, 0>>>(
            xgeom.data(), thrust::raw_pointer_cast(G_entity.data()),
            geometry_dofmap.data(),
            thrust::raw_pointer_cast(dphi_geometry.data()),
            thrust::raw_pointer_cast(Gweights_d.data()), cell_list_d.data(),
            cell_list_d.size());
        spdlog::debug("Done geometry_computation");
      }
    }
    else
      throw std::runtime_error("Unsupported degree [geometry]");
  }

  /// Compute matrix diagonal entries
  template <int P, int Q, typename Vector>
  void compute_mat_diag_inv(Vector& out)
  {
    T* geometry_ptr = thrust::raw_pointer_cast(G_entity.data());

    if (!lcells_device.empty())
    {
      spdlog::debug("mat_diagonal doing lcells. lcells size = {}",
                    lcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(lcells_device.data()),
                                 lcells_device.size());

      if (batch_size > 0)
      {
        spdlog::debug("Calling compute_geometry on local cells [{}]",
                      cell_list_d.size());
        compute_geometry(Q, cell_list_d);
        device_synchronize();
      }

      out.set(T{0.0});
      T* y = out.mutable_array().data();

      dim3 block_size(P + 1, P + 1, P + 1);
      dim3 grid_size(cell_list_d.size());
      spdlog::debug("Calling mat_diagonal");
      mat_diagonal<T, P, Q><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
          cell_list_d.data(), cell_list_d.size(), bc_marker.data());
      check_device_last_error();
    }

    if (!bcells_device.empty())
    {
      spdlog::debug("mat_diagonal doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());

      if (batch_size > 0)
      {
        compute_geometry(Q, cell_list_d);
        device_synchronize();
      }
      else
        geometry_ptr += 6 * Q * Q * Q * lcells_device.size();

      T* y = out.mutable_array().data();

      dim3 block_size(P + 1, P + 1, P + 1);
      dim3 grid_size(cell_list_d.size());
      mat_diagonal<T, P, Q><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
          cell_list_d.data(), cell_list_d.size(), bc_marker.data());
      check_device_last_error();
    }

    // Invert
    thrust::transform(thrust::device, out.array().begin(),
                      out.array().begin() + out.map()->size_local(),
                      out.mutable_array().begin(),
                      [] __host__ __device__(T yi) { return 1.0 / yi; });
  }

  template <int P, int Q, typename Vector>
  void impl_operator(Vector& in, Vector& out)
  {
    spdlog::debug("impl_operator operator start");

    in.scatter_fwd_begin();

    T* geometry_ptr = thrust::raw_pointer_cast(G_entity.data());

    if (!lcells_device.empty())
    {
      std::size_t i = 0;
      std::size_t i_batch_size
          = (batch_size == 0) ? lcells_device.size() : batch_size;
      while (i < lcells_device.size())
      {
        std::size_t i_next = std::min(lcells_device.size(), i + i_batch_size);
        std::span<int> cell_list_d(
            thrust::raw_pointer_cast(lcells_device.data()) + i, (i_next - i));
        i = i_next;

        if (batch_size > 0)
        {
          spdlog::debug("Calling compute_geometry on local cells [{}]",
                        cell_list_d.size());
          compute_geometry(Q, cell_list_d);
          device_synchronize();
        }

        spdlog::debug("Calling stiffness_operator on local cells [{}]",
                      cell_list_d.size());
        T* x = in.mutable_array().data();
        T* y = out.mutable_array().data();

        dim3 block_size(Q, Q, Q);
        dim3 grid_size(cell_list_d.size());
        stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
            x, cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
            cell_list_d.data(), cell_list_d.size(), bc_marker.data(),
            is_identity);

        check_device_last_error();
      }
    }

    spdlog::debug("impl_operator done lcells");

    spdlog::debug("cell_constants size {}", cell_constants.size());
    spdlog::debug("in size {}", in.array().size());
    spdlog::debug("out size {}", out.array().size());
    spdlog::debug("G_entity size {}", G_entity.size());
    spdlog::debug("cell_dofmap size {}", cell_dofmap.size());
    spdlog::debug("bc_marker size {}", bc_marker.size());

    in.scatter_fwd_end();

    spdlog::debug("impl_operator after scatter");

    if (!bcells_device.empty())
    {
      spdlog::debug("impl_operator doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());

      if (batch_size > 0)
      {
        compute_geometry(Q, cell_list_d);
        device_synchronize();
      }
      else
        geometry_ptr += 6 * Q * Q * Q * lcells_device.size();

      T* x = in.mutable_array().data();
      T* y = out.mutable_array().data();

      dim3 block_size(Q, Q, Q);
      dim3 grid_size(cell_list_d.size());
      stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
          x, cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
          cell_list_d.data(), cell_list_d.size(), bc_marker.data(),
          is_identity);

      check_device_last_error();
    }

    device_synchronize();

    spdlog::debug("impl_operator done bcells");
  }

  /// @brief Apply Laplacian operator
  /// @param in Input vector
  /// @param out Output vector
  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    out.set(T{0.0});

    if (op_nq == degree + 1)
    {
      if (degree == 1)
        impl_operator<1, 2>(in, out);
      else if (degree == 2)
        impl_operator<2, 3>(in, out);
      else if (degree == 3)
        impl_operator<3, 4>(in, out);
      else if (degree == 4)
        impl_operator<4, 5>(in, out);
      else if (degree == 5)
        impl_operator<5, 6>(in, out);
      else if (degree == 6)
        impl_operator<6, 7>(in, out);
      else if (degree == 7)
        impl_operator<7, 8>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else if (op_nq == degree + 2)
    {
      if (degree == 1)
        impl_operator<1, 3>(in, out);
      else if (degree == 2)
        impl_operator<2, 4>(in, out);
      else if (degree == 3)
        impl_operator<3, 5>(in, out);
      else if (degree == 4)
        impl_operator<4, 6>(in, out);
      else if (degree == 5)
        impl_operator<5, 7>(in, out);
      else if (degree == 6)
        impl_operator<6, 8>(in, out);
      else if (degree == 7)
        impl_operator<7, 9>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else
    {
      throw std::runtime_error("Unsupported nq");
    }

    spdlog::debug("Mat free operator end");
  }

  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    spdlog::debug("Mat diagonal operator start");

    if (op_nq == degree + 1)
    {
      if (degree == 1)
        compute_mat_diag_inv<1, 2>(diag_inv);
      else if (degree == 2)
        compute_mat_diag_inv<2, 3>(diag_inv);
      else if (degree == 3)
        compute_mat_diag_inv<3, 4>(diag_inv);
      else if (degree == 4)
        compute_mat_diag_inv<4, 5>(diag_inv);
      else if (degree == 5)
        compute_mat_diag_inv<5, 6>(diag_inv);
      else if (degree == 6)
        compute_mat_diag_inv<6, 7>(diag_inv);
      else if (degree == 7)
        compute_mat_diag_inv<7, 8>(diag_inv);
      else if (degree == 8)
        compute_mat_diag_inv<8, 9>(diag_inv);
      else if (degree == 9)
        compute_mat_diag_inv<9, 10>(diag_inv);
      else if (degree == 10)
        compute_mat_diag_inv<10, 11>(diag_inv);
      else
        throw std::runtime_error("Unsupported degree [mat diag]");
    }
    else if (op_nq == degree + 2)
    {
      if (degree == 1)
        compute_mat_diag_inv<1, 3>(diag_inv);
      else if (degree == 2)
        compute_mat_diag_inv<2, 4>(diag_inv);
      else if (degree == 3)
        compute_mat_diag_inv<3, 5>(diag_inv);
      else if (degree == 4)
        compute_mat_diag_inv<4, 6>(diag_inv);
      else if (degree == 5)
        compute_mat_diag_inv<5, 7>(diag_inv);
      else if (degree == 6)
        compute_mat_diag_inv<6, 8>(diag_inv);
      else if (degree == 7)
        compute_mat_diag_inv<7, 9>(diag_inv);
      else if (degree == 8)
        compute_mat_diag_inv<8, 10>(diag_inv);
      else if (degree == 9)
        compute_mat_diag_inv<9, 11>(diag_inv);
      else if (degree == 10)
        compute_mat_diag_inv<10, 12>(diag_inv);
      else
        throw std::runtime_error("Unsupported degree [mat diag]");
    }
    else
      throw std::runtime_error("Unsupported qmode [mat diag]");

    spdlog::debug("Mat diagonal operator end");
  }

private:
  int degree;

  // Number of quadrature points in 1D
  int op_nq;

  // Reference to on-device storage for constants, dofmap etc.
  std::span<const T> cell_constants;
  std::span<const std::int32_t> cell_dofmap;

  // Reference to on-device storage of geometry data
  std::span<const T> xgeom;
  std::span<const std::int32_t> geometry_dofmap;

  // geometry tables dphi on device
  thrust::device_vector<T> dphi_geometry;

  std::span<const std::int8_t> bc_marker;

  // On device storage for geometry quadrature weights
  thrust::device_vector<T> Gweights_d;

  // On device storage for geometry data (computed for each batch of cells)
  thrust::device_vector<T> G_entity;

  // Interpolation is the identity
  bool is_identity;

  // Lists of cells which are local (lcells) and boundary (bcells)
  thrust::device_vector<int> lcells_device, bcells_device;

  // On device storage for the inverse diagonal, needed for Jacobi
  // preconditioner (to remove in future)
  thrust::device_vector<T> _diag_inv;

  // Batch size for geometry computation (set to 0 for no batching)
  std::size_t batch_size;

  template <int P, int Q>
  void copy_phi_tables(std::span<const T> phi0, std::span<const T> dphi1)
  {
    err_check(deviceMemcpyToSymbol((phi0_const<T, P, Q>), phi0.data(),
                                   phi0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((dphi1_const<T, Q>),
                                   dphi1.data() + dphi1.size() / 2,
                                   (dphi1.size() / 2) * sizeof(T)));
  }
};
} // namespace benchdolfinx

#endif
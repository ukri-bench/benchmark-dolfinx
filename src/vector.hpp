// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/dolfinx_la.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <type_traits>

namespace benchdolfinx::impl
{
/// @brief pack data before MPI (neighbor) all-to-all operation
/// @tparam T Scalar data type
/// @param N Number of entries in indices
/// @param indices Indices of input data to be packed
/// @param in Input data to be sent: from owned region, for forward scatter, or
/// from ghost region for reverse scatter.
/// @param out Output data packed into blocks for each receiving MPI process
template <typename T>
static __global__ void pack_gpu(const int N,
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
static __global__ void unpack_gpu(const int N,
                                  const std::int32_t* __restrict__ indices,
                                  const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    out[indices[gid]] = in[gid];
  }
}

/// @brief unpack data after MPI all-to-all operation
/// @tparam T Scalar data type
/// @param N Number of entries in indices
/// @param indices Indices of output data to be unpacked into
/// @param in Input data "ghost" region
/// @param out Output data "owned" values
/// @note Accumulates values if multiple are received with the same index,
/// should be used for reverse scatter to owned region.
template <typename T>
static __global__ void
unpack_add_gpu(std::int32_t N, const int32_t* __restrict__ indices,
               const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    atomicAdd(&out[indices[gid]], in[gid]);
  }
}
} // namespace benchdolfinx::impl

namespace benchdolfinx
{

template <typename T>
using Vector = dolfinx::la::Vector<T, thrust::device_vector<T>,
                                   thrust::device_vector<std::int32_t>>;

/// @brief Get the pack function for a vector type.
///
/// out[i] = in[idx[i]]
template <typename T>
auto get_pack_fn(int block_size)
{
  return
      [block_size](
          typename thrust::device_vector<std::int32_t>::const_iterator
              idx_first,
          typename thrust::device_vector<std::int32_t>::const_iterator idx_last,
          typename thrust::device_vector<T>::const_iterator in_first,
          typename thrust::device_vector<T>::iterator out_first)
  {
    if (std::size_t d = thrust::distance(idx_first, idx_last); d > 0)
    {
      int num_blocks = (d + block_size - 1) / block_size;
      dim3 dim_block(block_size);
      dim3 dim_grid(num_blocks);

      const int32_t* idx_ptr = thrust::raw_pointer_cast(&idx_first[0]);
      const T* in_ptr = thrust::raw_pointer_cast(&in_first[0]);
      T* out_ptr = thrust::raw_pointer_cast(&out_first[0]);
      benchdolfinx::impl::pack_gpu<T>
          <<<dim_grid, dim_block, 0, 0>>>(d, idx_ptr, in_ptr, out_ptr);
      device_synchronize();
    }
  };
};

/// @brief
///
/// out[idx[i]] = in[i]
template <typename T>
auto get_unpack_fn(int block_size, int num_blocks)
{
  return
      [num_blocks, block_size](
          typename thrust::device_vector<std::int32_t>::const_iterator
              idx_first,
          typename thrust::device_vector<std::int32_t>::const_iterator idx_last,
          typename thrust::device_vector<T>::const_iterator in_first,
          typename thrust::device_vector<T>::iterator out_first)
  {
    dim3 dim_block(block_size);
    dim3 dim_grid(num_blocks);
    spdlog::debug("scatter_fwd_end step 2");
    if (std::size_t d = thrust::distance(idx_first, idx_last); d > 0)
    {
      const int32_t* idx_ptr = thrust::raw_pointer_cast(&idx_first[0]);
      const T* in_ptr = thrust::raw_pointer_cast(&in_first[0]);
      T* out_ptr = thrust::raw_pointer_cast(&out_first[0]);
      benchdolfinx::impl::unpack_gpu<T>
          <<<dim_grid, dim_block, 0, 0>>>(d, idx_ptr, in_ptr, out_ptr);
      device_synchronize();
    }
  };
};

/// @brief Compute the inner product of two vectors.
///
/// The two vectors must have the same parallel layout.
///
/// @note Collective MPI operation.
/// @param a A vector.
/// @param b A vector.
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename U>
auto inner_product(const U& a, const U& b)
{
  using T = typename U::value_type;

  const std::int32_t local_size = a.bs() * a.index_map()->size_local();
  if (local_size != b.bs() * b.index_map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");

  T local = thrust::inner_product(thrust::device, a.array().begin(),
                                  std::next(a.array().begin(), local_size),
                                  b.array().begin(), T{0});

  T result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM,
                a.index_map()->comm());
  return result;
}

/// @brief Compute the squared L2 norm of vector.
///
/// @param a Vector input
/// @note Collective MPI operation
template <typename U>
auto squared_norm(const U& a)
{
  using T = typename U::value_type;
  T result = benchdolfinx::inner_product(a, a);
  return std::real(result);
}

/// @brief Compute the norm of the vector.
///
/// @note Collective MPI operation
/// @param a A vector
/// @param type Norm type (supported types are \f$L^2\f$ and
/// \f$L^\infty\f$).
template <typename U>
auto norm(const U& a, dolfinx::la::Norm type = dolfinx::la::Norm::l2)
{
  switch (type)
  {
  case dolfinx::la::Norm::l2:
    return std::sqrt(benchdolfinx::squared_norm(a));
  case dolfinx::la::Norm::linf:
  {
    const std::int32_t size_local = a.bs() * a.index_map()->size_local();
    auto max_pos
        = thrust::max_element(thrust::device, a.array().begin(),
                              thrust::next(a.array().begin(), size_local));
    auto local_linf = std::abs(*max_pos);
    decltype(local_linf) linf = 0;
    MPI_Allreduce(&local_linf, &linf, 1, dolfinx::MPI::mpi_t<decltype(linf)>,
                  MPI_MAX, a.index_map()->comm());
    return linf;
  }
  default:
    throw std::runtime_error("Norm type not supported");
  }
}

/// @brief Compute vector r = alpha * x + y.
///
/// @tparam S Scalar Type
/// @tparam Vector Vector Type
/// @param [in,out] r Result
/// @param alpha
/// @param [in] x
/// @param [in] y
template <typename U, typename S>
void axpy(U& r, S alpha, const U& x, const U& y)
{
  spdlog::debug("AXPY start");
  using T = typename U::value_type;
  thrust::transform(
      thrust::device, x.array().begin(),
      thrust::next(x.array().begin(), x.index_map()->size_local()),
      y.array().begin(), r.array().begin(),
      [alpha] __host__ __device__(const T& vx, const T& vy)
      { return vx * alpha + vy; });
  spdlog::debug("AXPY end");
}

/// @brief Scale vector by alpha.
/// @param [in,out] r Result
/// @param alpha
template <typename U, typename S>
void scale(U& r, S alpha)
{
  using T = typename U::value_type;
  thrust::for_each(thrust::device, r.array().begin(), r.array().end(),
                   [alpha] __host__ __device__(T & v) { v *= alpha; });
}

/// @brief Copy Vector b into Vector a.
///
/// @param [in,out] a
/// @param [in] b
/// @note Only copies the owned part of the Vector, no ghosts
/// @note a must be the same size as b
template <typename U>
void copy(U& a, const U& b)
{
  std::int32_t local_size = a.bs() * a.index_map()->size_local();
  thrust::copy_n(thrust::device, b.begin(), local_size, a.begin());
}

/// @brief Compute pointwise vector multiplication w[i] = x[i] * y[i].
///
/// @param [in,out] w
/// @param [in] x
/// @param [in] y
/// @note w, x, and y must all be the same size
/// @note Only computes on the owned part of the Vector, no ghosts
template <typename U>
void pointwise_mult(U& w, const U& x, const U& y)
{
  spdlog::debug("pointwise_mult start");

  using T = typename U::value_type;
  thrust::transform(
      thrust::device, x.array().begin(),
      thrust::next(x.array().begin(), x.index_map()->size_local()),
      y.array().begin(), w.array().begin(),
      [] __host__ __device__(T xi, T yi) { return xi * yi; });
  spdlog::debug("pointwise_mult end");
}

/// TODO
template <typename U, typename T>
void set_value(U& x, T v)
{
  thrust::fill(x.array().begin(), x.array().end(), v);
}

} // namespace benchdolfinx

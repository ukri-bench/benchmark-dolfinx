// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include "vector.hpp"
#include <dolfinx/la/MatrixCSR.h>
#include <thrust/device_vector.h>

namespace benchdolfinx::impl
{
/// @brief Computes y += A*x for a local CSR matrix A and local dense
/// vectors x,y.
///
/// @param N number of rows
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values
/// and indices.
/// @param[in] row_end Last index of each row in the arrays values and
/// indices.
/// @param[in] indices Column indices for each non-zero element of the
/// matrix A
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
__global__ void spmv_impl(int N, const T* values, const std::int32_t* row_begin,
                          const std::int32_t* row_end,
                          const std::int32_t* indices, const T* x, T* y)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the row index is out of bounds.
  if (i < N)
  {
    // Perform the sparse matrix-vector multiplication for this row.
    T vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

/// @brief Computes y += A^T*x for a local CSR matrix A and local dense
/// vectors x,y.
//
/// @param N number of rows
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values
/// and indices.
/// @param[in] row_end Last index of each row in the arrays values and
/// indices.
/// @param[in] indices Column indices for each non-zero element of the
/// matrix A
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
__global__ void spmvT_impl(int N, const T* values,
                           const std::int32_t* row_begin,
                           const std::int32_t* row_end,
                           const std::int32_t* indices, const T* x, T* y)
{
  // Calculate the row index for this thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the row index is out of bounds
  if (i < N)
  {
    // Perform the transpose sparse matrix-vector multiplication for
    // this row
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      atomicAdd(&y[indices[j]], values[j] * x[i]);
  }
}

template <typename T>
thrust::device_vector<T>
compute_diag_inv(dolfinx::la::MatrixCSR<T, thrust::device_vector<T>,
                                        thrust::device_vector<std::int32_t>,
                                        thrust::device_vector<std::int32_t>>& A)
{
  thrust::device_vector<T> diag_inv(A.index_map(0)->size_local());
  for (int i = 0; i < static_cast<int>(diag_inv.size()); ++i)
  {
    // Find diagonal entry on each row
    thrust::copy_if(thrust::device,
                    thrust::next(A.values().begin(), A.row_ptr()[i]),
                    thrust::next(A.values().begin(), A.row_ptr()[i + 1]),
                    thrust::next(A.cols().begin(), A.row_ptr()[i]),
                    thrust::next(diag_inv.begin(), i),
                    [=] __host__ __device__(std::int32_t col) -> bool
                    { return (col == i); });
  }

  thrust::transform(thrust::device, diag_inv.begin(), diag_inv.end(),
                    diag_inv.begin(),
                    [] __host__ __device__(T x) { return 1 / x; });
  return diag_inv;
}

} // namespace benchdolfinx::impl

namespace benchdolfinx
{
/// @brief An assembled matrix operator for a finite element Form,
/// internally using a CSR matrix.
/// @tparam T
template <typename T>
class MatrixOperator
{
public:
  /// The value type
  using value_type = T;

  /// @brief An on-device CSR Matrix
  /// @param A CPU-based CSR Matrix
  MatrixOperator(
      dolfinx::la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>,
                             std::vector<std::int32_t>>& A)
      : _A(A)
  {
    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    // Copy to device
    //    _A = std::make_unique<dolfinx::la::MatrixCSR<
    //        T, thrust::device_vector<T>, thrust::device_vector<std::int32_t>,
    //        thrust::device_vector<std::int32_t>>>(A);

    spdlog::info("A norm = {}", norm());

    // Get inverse diagonal entries (for Jacobi preconditioning)
    _diag_inv = impl::compute_diag_inv<T>(_A);

    spdlog::info("Created device CSR matrix");
  }

  /// Destructor
  ~MatrixOperator() = default;

  /// Compute Matrix Norm
  /// @returns the Frobenius norm of the local part of the CSR matrix
  T norm()
  {
    T n0 = thrust::transform_reduce(
        thrust::device, _A.values().begin(), _A.values().end(),
        [] __host__ __device__(T x) -> T { return x * x; }, T(0.0),
        [] __host__ __device__(T x, T y) -> T { return x + y; });
    return std::sqrt(n0);
  }

  /// @brief Get the inverse of the diagonal values of the matrix.
  /// @param diag_inv [in/out] A Vector to copy the inverse diagonal
  /// values into.
  /// @note Vector must be the correct size.
  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    thrust::copy(_diag_inv.begin(), _diag_inv.end(), diag_inv.array().begin());
  }

  /// @brief The matrix-vector multiplication operator, y=Ax,
  /// multiplying the matrix with the input vector and stores the result
  /// in the output vector.
  ///
  /// @tparam Vector The type of the input and output vector.
  ///
  /// @param x The input vector. See note.
  /// @param y The output vector.
  /// @param transpose If true, perform the transpose operation y=A^T x
  /// @note x is not const because a scatter_fwd is done on it
  template <typename Vector>
  void apply(Vector& x, Vector& y, bool transpose = false)
  {
    dolfinx::common::Timer t0("% MatrixOperator application");

    set_value(y, T{0});
    T* _x = x.array().data().get();
    T* _y = y.array().data().get();

    if (transpose)
    {
      int num_rows = _A.index_map(0)->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin(get_pack_fn<T>(512),
                          [](auto&& x) { return x.data().get(); });
      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _A.values().data().get(), _A.row_ptr().data().get(),
          _A.off_diag_offset().data().get(), _A.cols().data().get(), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end(get_unpack_fn<T>(512, 1));

      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _A.values().data().get(), _A.off_diag_offset().data().get(),
          _A.row_ptr().data().get() + 1, _A.cols().data().get(), _x, _y);
      check_device_last_error();
    }
    else
    {
      int num_rows = _A.index_map(0)->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin(get_pack_fn<T>(512),
                          [](auto&& x) { return x.data().get(); });
      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _A.values().data().get(), _A.row_ptr().data().get(),
          _A.off_diag_offset().data().get(), _A.cols().data().get(), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end(get_unpack_fn<T>(512, 1));

      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _A.values().data().get(), _A.off_diag_offset().data().get(),
          _A.row_ptr().data().get() + 1, _A.cols().data().get(), _x, _y);
      check_device_last_error();
    }

    device_synchronize();
  }

private:
  // CSR matrix in GPU memory
  dolfinx::la::MatrixCSR<T, thrust::device_vector<T>,
                         thrust::device_vector<std::int32_t>,
                         thrust::device_vector<std::int32_t>>
      _A;

  // Copy of the inverse of the diagonal entries of the matrix - may be
  // used for Jacobi preconditioning.
  thrust::device_vector<T> _diag_inv;
};
} // namespace benchdolfinx

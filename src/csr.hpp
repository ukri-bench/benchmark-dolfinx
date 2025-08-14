// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include <dolfinx/la/MatrixCSR.h>
#include <mpi.h>
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
__global__ void spmv_impl(int N, const thrust::device_vector<T>& values,
                          const std::int32_t* row_begin,
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
__global__ void spmvT_impl(int N, const thrust::device_vector<T>& values,
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

  /// @brief Construct a CSR matrix for the Form a, with given boundary
  /// conditions
  /// @param a A finite element form
  /// @param bcs Set of boundary conditions to be applied

  /// @brief Construct a CSR matrix for the Form a, with given boundary
  /// conditions.
  ///
  /// @param A DOLFINx sparse matrix.
  /// @param comm MPI communicator that the matrix is defined on.
  MatrixOperator(const dolfinx::la::MatrixCSR<T>& A, MPI_Comm comm)
      : _comm(comm), _row_map(A.index_map(0)), _col_map(A.index_map(1))
  {
    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    std::int32_t num_rows = _row_map->size_local();
    std::int32_t nnz = A.row_ptr()[num_rows];
    _nnz = nnz;

    T norm = 0;
    for (T v : A.values())
      norm += v * v;

    spdlog::info("A norm = {}", std::sqrt(norm));

    // Get inverse diagonal entries (for Jacobi preconditioning)
    std::vector<T> diag_inv(num_rows);
    for (int i = 0; i < num_rows; ++i)
    {
      for (int j = A.row_ptr()[i]; j < A.row_ptr()[i + 1]; ++j)
      {
        if (A.cols()[j] == i)
          diag_inv[i] = 1 / A.values()[j];
      }
    }

    _diag_inv = thrust::device_vector<T>(diag_inv.size());
    thrust::copy(diag_inv.begin(), diag_inv.end(), _diag_inv.begin());

    _row_ptr = thrust::device_vector<std::int32_t>(num_rows + 1);
    _off_diag_offset = thrust::device_vector<std::int32_t>(num_rows);
    _cols = thrust::device_vector<std::int32_t>(nnz);
    _values = thrust::device_vector<T>(nnz);

    // Copy data from host to device
    spdlog::info("Creating Device matrix with {} non zeros", _nnz);
    spdlog::info("Creating row_ptr with {} to {}", num_rows + 1,
                 _row_ptr.size());
    thrust::copy(A.row_ptr().begin(), A.row_ptr().begin() + num_rows + 1,
                 _row_ptr.begin());

    spdlog::info("Creating off_diag with {} to {}", A.off_diag_offset().size(),
                 _off_diag_offset.size());
    thrust::copy(A.off_diag_offset().begin(),
                 A.off_diag_offset().begin() + num_rows,
                 _off_diag_offset.begin());

    spdlog::info("Creating cols with {} to {}", nnz, _cols.size());
    thrust::copy(A.cols().begin(), A.cols().begin() + nnz, _cols.begin());
    spdlog::info("Creating values with {} to {}", nnz, _values.size());
    thrust::copy(A.values().begin(), A.values().begin() + nnz, _values.begin());
  }

  /// Destructor
  ~MatrixOperator() = default;

  /// @brief Get the inverse of the diagonal values of the matrix.
  /// @param diag_inv [in/out] A Vector to copy the inverse diagonal
  /// values into.
  /// @note Vector must be the correct size.
  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    thrust::copy(_diag_inv.begin(), _diag_inv.end(),
                 diag_inv.mutable_array().begin());
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

    y.set(0);
    T* _x = x.mutable_array().data();
    T* _y = y.mutable_array().data();

    if (transpose)
    {
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _values, thrust::raw_pointer_cast(_row_ptr.data()),
          thrust::raw_pointer_cast(_off_diag_offset.data()),
          thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end();

      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _values, thrust::raw_pointer_cast(_off_diag_offset.data()),
          thrust::raw_pointer_cast(_row_ptr.data()) + 1,
          thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
    }
    else
    {
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _values, thrust::raw_pointer_cast(_row_ptr.data()),
          thrust::raw_pointer_cast(_off_diag_offset.data()),
          thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end();

      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _values, thrust::raw_pointer_cast(_off_diag_offset.data()),
          thrust::raw_pointer_cast(_row_ptr.data()) + 1,
          thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
    }

    device_synchronize();
  }

  /// @brief IndexMap for the column space of the matrix
  /// @returns IndexMap
  std::shared_ptr<const dolfinx::common::IndexMap> column_index_map()
  {
    return _col_map;
  }

  /// @brief IndexMap for the row space of the matrix
  /// @returns IndexMap
  std::shared_ptr<const dolfinx::common::IndexMap> row_index_map()
  {
    return _row_map;
  }

  /// @brief Number of non-zeros in the matrix
  /// @returns Number of non-zeros
  std::size_t nnz() { return _nnz; }

private:
  // Number of non-zeros in the matrix
  std::size_t _nnz;

  // Values stored on-device using CSR storage, _values, _cols and
  // _row_ptr, as conventional for CSR
  thrust::device_vector<T> _values;
  thrust::device_vector<std::int32_t> _cols;
  thrust::device_vector<std::int32_t> _row_ptr;

  // Values stored on-device.
  // Copy of the inverse of the diagonal entries of the matrix - may be
  // used for Jacobi preconditioning.
  thrust::device_vector<T> _diag_inv;

  // Start point, on each row, of the off-diagonal block (ghost region)
  thrust::device_vector<std::int32_t> _off_diag_offset;

  // IndexMaps for the columns and rows of the matrix
  std::shared_ptr<const dolfinx::common::IndexMap> _col_map, _row_map;

  // MPI Comm associated with the Matrix
  MPI_Comm _comm;
};
} // namespace benchdolfinx

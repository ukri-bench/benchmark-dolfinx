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
  MatrixOperator(
      const dolfinx::fem::Form<T, T>& a,
      std::vector<std::reference_wrapper<const dolfinx::fem::DirichletBC<T>>>
          bcs)
  {
    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    if (a.rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    auto V = a.function_spaces()[0];
    dolfinx::la::SparsityPattern pattern
        = dolfinx::fem::create_sparsity_pattern(a);
    pattern.finalize();

    // Assemble on CPU
    dolfinx::la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>,
                           std::vector<std::int32_t>>
        A(pattern);
    dolfinx::fem::assemble_matrix(A.mat_add_values(), a, bcs);
    A.scatter_rev();
    dolfinx::fem::set_diagonal<T>(A.mat_set_values(), *V, bcs, T(1.0));

    // Copy to device
    _A = std::make_unique<dolfinx::la::MatrixCSR<
        T, thrust::device_vector<T>, thrust::device_vector<std::int32_t>,
        thrust::device_vector<std::int32_t>>>(A);

    // Compute Matrix Norm
    T norm = thrust::transform_reduce(
        thrust::device, _A->values().begin(), _A->values().end(),
        [] __device__(auto x) { return x * x; }, T(0.0),
        [] __device__(auto x, auto y) { return x + y; });
    spdlog::info("A norm = {}", std::sqrt(norm));

    // Get inverse diagonal entries (for Jacobi preconditioning)
    _diag_inv = thrust::device_vector<T>(_A->index_map(0)->size_local());
    for (int i = 0; i < _diag_inv.size(); ++i)
    {
      // Find diagonal entry on each row
      thrust::copy_if(thrust::device,
                      thrust::next(_A->values().begin(), _A->row_ptr()[i]),
                      thrust::next(_A->values().begin(), _A->row_ptr()[i + 1]),
                      thrust::next(_A->cols().begin(), _A->row_ptr()[i]),
                      thrust::next(_diag_inv.begin(), i),
                      [=] __device__(auto col) { return (col == i); });
    }
    thrust::transform(thrust::device, _diag_inv.begin(), _diag_inv.end(),
                      _diag_inv.begin(), [](auto x) { return 1 / x; });

    spdlog::info("Created device CSR matrix");
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
      int num_rows = _A->index_map(0)->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin(get_pack_fn<T>(512),
                          [](auto&& x) { return x.data().get(); });
      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, _A->values().data().get(), _A->row_ptr().data().get(),
          _A->off_diag_offset().data().get(), _A->cols().data().get(), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end(get_unpack_fn<T>(512, 1));

      impl::spmvT_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, thrust::raw_pointer_cast(_A->values().data()),
          thrust::raw_pointer_cast(_A->off_diag_offset().data()),
          thrust::raw_pointer_cast(_A->row_ptr().data()) + 1,
          thrust::raw_pointer_cast(_A->cols().data()), _x, _y);
      check_device_last_error();
    }
    else
    {
      int num_rows = _A->index_map(0)->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin(get_pack_fn<T>(512),
                          [](auto&& x) { return x.data().get(); });
      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, thrust::raw_pointer_cast(_A->values().data()),
          thrust::raw_pointer_cast(_A->row_ptr().data()),
          thrust::raw_pointer_cast(_A->off_diag_offset().data()),
          thrust::raw_pointer_cast(_A->cols().data()), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end(get_unpack_fn<T>(512, 1));

      impl::spmv_impl<T><<<grid_size, block_size, 0, 0>>>(
          num_rows, thrust::raw_pointer_cast(_A->values().data()),
          thrust::raw_pointer_cast(_A->off_diag_offset().data()),
          thrust::raw_pointer_cast(_A->row_ptr().data()) + 1,
          thrust::raw_pointer_cast(_A->cols().data()), _x, _y);
      check_device_last_error();
    }

    device_synchronize();
  }

private:
  // CSR matrix in GPU memory
  std::unique_ptr<dolfinx::la::MatrixCSR<T, thrust::device_vector<T>,
                                         thrust::device_vector<std::int32_t>,
                                         thrust::device_vector<std::int32_t>>>
      _A;

  // Copy of the inverse of the diagonal entries of the matrix - may be
  // used for Jacobi preconditioning.
  thrust::device_vector<T> _diag_inv;
};
} // namespace benchdolfinx

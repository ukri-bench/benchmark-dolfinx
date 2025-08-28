// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace benchdolfinx
{
/// @brief Computes weighted geometry tensor G from the coordinates and
/// quadrature weights.
///
/// @param[in] xgeom Geometry points [:, 3]
/// @param[out] G_entity geometry data [n_entities, nq, 6]
/// @param[in] geometry_dofmap Location of coordinates for each cell in
/// `xgeom [:, ncdofs]`.
/// @param[in] dphi Basis derivative tabulation for cell at quadrature
/// `points [3, nq, ncdofs]`.
/// @param[in] weights Quadrature weights [nq]
/// @param[in] entities list of cells to compute for [n_entities]
/// @param[in] n_entities total number of cells to compute for
/// @tparam T scalar type
/// @tparam Q number of quadrature points (in 1D)
template <typename T, int Q>
__global__ void geometry_computation_gpu(const T* xgeom, T* G_entity,
                                         const std::int32_t* geometry_dofmap,
                                         const T* dphi, const T* weights,
                                         const int* entities, int n_entities)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of quadrature points (must match arrays in weights and dphi)
  constexpr int nq = Q * Q * Q;

  // Number of coordinate dofs
  constexpr int ncdofs = 8;

  // Geometric dimension
  constexpr int gdim = 3;

  __shared__ T shared_mem[ncdofs * gdim];

  // coord_dofs has shape [ncdofs, gdim]
  T* coord_dofs = shared_mem;

  // Bring cell geometry into shared memory
  int iq = threadIdx.x;
  if constexpr (nq < 27)
  {
    // Only 8 threads when Q == 2
    assert(iq < 8);
    for (int j = 0; j < 3; ++j)
    {
      coord_dofs[iq * 3 + j]
          = xgeom[3 * geometry_dofmap[cell * ncdofs + iq] + j];
    }
  }
  else
  {
    int i = iq / gdim;
    int j = iq % gdim;
    if (i < ncdofs)
      coord_dofs[iq] = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }

  __syncthreads();
  // One quadrature point per thread

  if (iq >= nq)
    return;

  // Jacobian
  T J[3][3];
  auto idx = [](int i, int j) { return i * gdim + j; };

  // For each quadrature point / thread
  {
    auto idx_dphi = [iq](int i, int j) { return (i * nq + iq) * ncdofs + j; };
    for (std::size_t i = 0; i < gdim; i++)
    {
      for (std::size_t j = 0; j < gdim; j++)
      {
        J[i][j] = 0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs[idx(k, i)] * dphi[idx_dphi(j, k)];
      }
    }

    // Components of K = J^-1 (detJ)
    T K[3][3] = {{J[1][1] * J[2][2] - J[1][2] * J[2][1],
                  -J[0][1] * J[2][2] + J[0][2] * J[2][1],
                  J[0][1] * J[1][2] - J[0][2] * J[1][1]},
                 {-J[1][0] * J[2][2] + J[1][2] * J[2][0],
                  J[0][0] * J[2][2] - J[0][2] * J[2][0],
                  -J[0][0] * J[1][2] + J[0][2] * J[1][0]},
                 {J[1][0] * J[2][1] - J[1][1] * J[2][0],
                  -J[0][0] * J[2][1] + J[0][1] * J[2][0],
                  J[0][0] * J[1][1] - J[0][1] * J[1][0]}};

    T detJ = J[0][0] * K[0][0] - J[0][1] * K[1][0] + J[0][2] * K[2][0];

    std::size_t offset = (c * nq * 6 + iq);
    G_entity[offset]
        = (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + nq]
        = (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 2 * nq]
        = (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 3 * nq]
        = (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 4 * nq]
        = (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 5 * nq]
        = (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2])
          * weights[iq] / detJ;
  }
}

} // namespace benchdolfinx

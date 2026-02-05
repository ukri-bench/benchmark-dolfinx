// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "geometry_gpu.hpp"
#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace benchdolfinx
{
/// @brief Compute 3d index from 1d indices.
///
/// Compute the index `idx = ld0 * i + ld1 * j + ld2 * k`.
///
/// For contiguous, row-major storage of a tensor with shape `(n0, n1,
/// n2)`, use `ld0=n1*n2`, `ld1=n2`, `ld2=1` (`k` varies fastest,
/// followed by `j`).
///
/// For contiguous, column-major storage of a tensor with shape `(n0,
/// n1, n2)`, use `ld0=1`, `ld1=n0`, `ld2=n0*n1` (`i` varies fastest,
/// followed by `j`).
///
/// For contiguous storage with `j` varying fastest and `i` slowest, use
/// `ld0=n1*n2`, `ld1=1`, `ld2=n1`
///
/// For contiguous storage with `j` varying fastest and `k` slowest, use
/// `ld0=n1`, `ld1=1`, `ld2=n0*n1`
///
/// @tparam ld0 Stride for first (`i`) index.
/// @tparam ld1 Stride for second (`k`) index.
/// @tparam ld2 Stride for third (`k`) index.
/// @param[in] i
/// @param[in] j
/// @param[in] k
/// @return Flattened index.
template <int ld0, int ld1, int ld2>
__device__ __forceinline__ int ijk(int i, int j, int k)
{
  return i * ld0 + j * ld1 + k * ld2;
}

/// Compute b = A * u where A is the stiffness operator for a set of
/// entities (cells or facets) in a mesh.
///
/// The stiffness operator is defined as:
///
///     A = ∑_i ∫_Ω C ∇ϕ_i ∇ϕ_j dx
///
/// where C is a constant, ϕ_i and ϕ_j are the basis functions of the
/// finite element space, and ∇ϕ_i is the gradient of the basis
/// function. The integral is computed over the domain Ω of the entity
/// using sum factorization. The basis functions are defined on the
/// reference element and are transformed to the physical element using
/// the geometry operator G. G is a 3x3 matrix per quadrature point per
/// entity.
///
/// @tparam T Data type of the input and output arrays
/// @tparam P Polynomial degree of the basis functions
/// @tparam Q Number of quadrature points in 1D
/// @param u Input vector of size (ndofs,)
/// @param entity_constants Array of size (n_entities,) with the
/// constant C for each entity
/// @param b Output vector of size (ndofs,)
/// @param G_entity Array of size (n_entities, nq, 6) with the geometry
/// operator G for each entity
/// @param entity_dofmap Array of size (n_entities, ndofs) with the
/// dofmap for each entity
/// @param phi0_in Array of size (nq, ndofs) with the interpolation basis
/// functions in 1D. u1_i = phi0_(ij) u_j, where u are the dofs
/// associated with the element (degree P), and u1 are the dofs for the
/// finite elment (degree >= P) that u is interpolated into.
/// @param dphi1_in Array of size (nq, nq) with the 1D basis function
/// derivatives. FIXME: layout is (point_idx, dphi_i)?
/// @param entities List of entities to compute on
/// @param n_entities Number of entries in `entities`
/// @param bc_marker Array of size (ndofs,) with the boundary condition
/// marker
/// @param identity If 1, the basis functions are the identity for the
/// given quadrature points
///
/// @note The kernel is launched with a 3D grid of 1D blocks, where each
/// block is responsible for computing the stiffness operator for a
/// single entity. The block size is (P+1, P+1, P+1) and the shared
/// memory 2 * (P+1)^3 * sizeof(T).
template <typename T, int P, int Q>
__launch_bounds__(Q * Q * Q) __global__ void stiffness_operator_gpu(
    const T* __restrict__ u, const T* __restrict__ entity_constants,
    T* __restrict__ b, const T* __restrict__ G_entity,
    const T* __restrict__ phi0_const, const T* __restrict__ dphi1_const,
    const std::int32_t* __restrict__ entity_dofmap,
    const int* __restrict__ entities, int n_entities,
    const std::int8_t* __restrict__ bc_marker, bool identity)
{
  // Note: each thread is respinsible for one quadrature point. Since
  // the number of DOFs is less than or equal to the number quadrature
  // points a subset of threads are also responsible for one DOF
  // contribiution (an entry in b).

  constexpr int nd = P + 1; // Number of dofs per direction in 1D
  constexpr int nq = Q;     // Number of quadrature points in 1D

  assert(blockDim.x == nq);
  assert(blockDim.y == nq);
  assert(blockDim.z == nq);

  // block_id is the cell (or facet) index
  const int block_id = blockIdx.x;

  // Check if the block_id is valid (i.e. within the number of entities)
  if (block_id >= n_entities) // Should always be true
    return;

  constexpr int square_nd = nd * nd;
  constexpr int square_nq = nq * nq;
  constexpr int cube_nd = square_nd * nd;
  constexpr int cube_nq = square_nq * nq;

  constexpr int nq1 = nq + 1;

  // Try padding
  __shared__ T scratch1[cube_nq];
  __shared__ T scratch2[cube_nq];
  __shared__ T scratch3[cube_nq];

  // Note: thread order on the device is such that
  // neighboring threads can get coalesced memory access, i.e.
  // tz threads are closest together (called .x by CUDA/HIP)
  const int tx = threadIdx.z; // 1d dofs x direction
  const int ty = threadIdx.y; // 1d dofs y direction
  const int tz = threadIdx.x; // 1d dofs z direction

  // thread_id represents the quadrature index in 3D ('row-major')
  const int thread_id = tx * square_nq + ty * nq + tz;

  // Copy phi and dphi to shared memory
  __shared__ T phi0[nq * nd];
  __shared__ T dphi1[nq1 * nq];

  if (thread_id < nd * nq)
    phi0[thread_id] = phi0_const[thread_id];

  if (tz < nq and ty < nq and tx == 0)
    dphi1[ty * nq1 + tz] = dphi1_const[ty * nq + tz];

  // Get dof value (in x) that this thread is responsible for, and
  // place in shared memory.
  int dof = -1;

  // Note: We might have more threads per block than dofs, so we need
  // to check if the thread_id is valid
  scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = 0;
  if (tx < nd && ty < nd && tz < nd)
  {
    int dof_thread_id = tx * square_nd + ty * nd + tz;
    int entity_index = entities[block_id];
    dof = entity_dofmap[entity_index * cube_nd + dof_thread_id];
    if (bc_marker[dof])
    {
      b[dof] = u[dof];
      dof = -1;
    }
    else
      scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = u[dof];
  }

  __syncthreads(); // Make sure all threads have written to shared memory

  // Interpolate basis functions to quadrature points
  if (identity != 1)
  {
    // Interpolate u from phi0 basis to phi1 basis. The phi1 basis nodes
    // are collcated with the quadrature points.
    //
    // Note: phi0 has shape (nq, nd)
    //
    // u(q0, q1, q2) = \phi0_(i)(q0) \phi0_(j)(q1) \phi0_(k)(q2) u_(i, j, k)
    //
    // 0. tmp0_(q0, j, k)  = \phi0_(i)(q0) u_(i, j, k)
    // 1. tmp1_(q0, q1, k) = \phi0_(j)(q1) tmp0_(q0, j, k)
    // 2. u_(q0, q1, q2)   = \phi0_(k)(q2) tmp1_(q0, q1, k)

    // 0. tmp0_(q0, j, k) = \phi_(i)(q0) u_(i, j, k)
    T xq = 0;
    for (int ix = 0; ix < nd; ++ix)
      xq += phi0[tx * nd + ix] * scratch2[ijk<square_nq, nq, 1>(ix, ty, tz)];

    scratch1[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();

    // 1. tmp1_(q0, q1, k) = \phi0_(j)(q1) tmp0_(q0, j, k)
    xq = 0;
    for (int iy = 0; iy < nd; ++iy)
      xq += phi0[ty * nd + iy] * scratch1[ijk<square_nq, nq, 1>(tx, iy, tz)];

    scratch3[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();

    // 2. u_(q0, q1, q2) = \phi_(k)(q2) tmp1_(q0, q1, k)
    xq = 0;
    for (int iz = 0; iz < nd; ++iz)
      xq += phi0[tz * nd + iz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, iz)];

    scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();
  } // end of interpolation

  // Compute du/dx0, du/dx1 and du/dx2 (deriavtives on reference cell)
  // at the quadrature point computed by this thread.
  //
  // From
  //
  //   u(q0, q1, q2) = \phi0_(i)(q0) \phi0_(j)(q1) \phi0_(k)(q2) u_(i, j, k)
  //
  // we have
  //
  //   (du/dx0)(q0, q1, q2) = (d\phi1_(i)/dx)(q0) \phi1_(j)(q1) \phi1_(k)(q2)
  //   u_(i, j, k) (du/dx1)(q0, q1, q2) = \phi1_(i)(q0) (d\phi1_(j)/dx)(q1)
  //   \phi1_(k)(q2) u_(i, j, k) (du/dx2)(q0, q1, q2) = \phi1_(i)(q0)
  //   \phi1_(j)(q1) (d\phi1_(k)/dx)(q1) u_(i, j, k)
  //
  // Quadrature points and the phi1 'degrees-of-freedom' are co-located,
  // therefore:
  //
  //   (du/dx0)(q0, q1, q2)
  //    = (d\phi1_(i)/dx)(q0) \phi1_(j)(q1) \phi1_(k)(q2) u_(i, j, k)
  //    = (d\phi1_(i)/dx)(q0) \delta_(q1, j) \delta_(q2, k) u_(i, j, k)
  //    = (d\phi1_i)/dx)_(q0) u_(i, q1, q2)
  //
  //   (du/dx1)(q0, q1, q2) = (d\phi1_(j)/dx)(q1) u_(q0, j, q2)
  //   (du/dx2)(q0, q1, q2) = (d\phi1_(j)/dx)(q2) u_(q1, q1, k)

  // (du/dx0)_(q0, q1, q2) = (d\phi1_(i)/dx)(q0) u_(i, q1, q2)
  T val_x = 0;
  for (int ix = 0; ix < nq; ++ix)
    val_x += dphi1[tx * nq1 + ix] * scratch2[ijk<square_nq, nq, 1>(ix, ty, tz)];

  // (du/dx1)_(q0, q1, q2) = (d\phi1_(j)/dx)(q1) u_(q0, j, q2)
  T val_y = 0;
  for (int iy = 0; iy < nq; ++iy)
    val_y += dphi1[ty * nq1 + iy] * scratch2[ijk<square_nq, nq, 1>(tx, iy, tz)];

  // (du/dx2)_(q0, q1, q2) = (d\phi1_(k)/dx)(q2) u_(q0, q1, k)
  T val_z = 0;
  for (int iz = 0; iz < nq; ++iz)
    val_z += dphi1[tz * nq1 + iz] * scratch2[ijk<square_nq, nq, 1>(tx, ty, iz)];

  // TODO: Add some maths

  // Apply geometric transformation to data at quadrature point
  const std::int64_t gid = static_cast<std::int64_t>(block_id) * cube_nq * 6 + thread_id;
  const T G0 = non_temp_load(&G_entity[gid + cube_nq * 0]);
  const T G1 = non_temp_load(&G_entity[gid + cube_nq * 1]);
  const T G2 = non_temp_load(&G_entity[gid + cube_nq * 2]);
  const T G3 = non_temp_load(&G_entity[gid + cube_nq * 3]);
  const T G4 = non_temp_load(&G_entity[gid + cube_nq * 4]);
  const T G5 = non_temp_load(&G_entity[gid + cube_nq * 5]);

  const T coeff = entity_constants[block_id];

  // Store values at quadrature points: scratch2, scratchy, scratchz all
  // have dimensions (nq, nq, nq)
  __syncthreads();
  int idx = ijk<square_nq, nq, 1>(tx, ty, tz);
  scratch1[idx] = coeff * (G0 * val_x + G1 * val_y + G2 * val_z);
  scratch2[idx] = coeff * (G1 * val_x + G3 * val_y + G4 * val_z);
  scratch3[idx] = coeff * (G2 * val_x + G4 * val_y + G5 * val_z);

  // Apply contraction in the x-direction
  // T grad_x = 0;
  // T grad_y = 0;
  // T grad_z = 0;

  // tx is dof index, ty, tz quadrature point indices

  // At this thread's quadrature point, compute r1 = r1_(i0, i1, i2) =
  // \sum_(q0, q2, q3) [\nabla\Phi1_(i0, i1, i2)](q0, q2, q3) \cdot [\nabla
  // u](q0, q2, q3)
  //
  // r1_(i0, i1, i2)
  //    = (d\Phi1_(i0, i1, i2)/dx0)(q0, q1, q2) (du/dx0)(q0, q1, q2)
  //    + (d\Phi1_(i0, i1, i2)/dx1)(q0, q1, q2) (du/dx1)(q0, q1, q2)
  //    + (d\Phi1_(i0, i1, i2)/dx2)(q0, q1, q2) (du/dx2)(q0, q1, q2)
  //
  //    = (d\phi1_(i0)/dx)(q0) \phi1_(i1)(q1) \phi1_(i2)(q2) (du/dx0)(q0, q1,
  //    q2)
  //    + \phi1_(i0)(q0) (d\phi1_(i1)/dx)(q1) \phi1_(i2)(q2) (du/dx1)(q0, q1,
  //    q2)
  //    + \phi1_(i0)(q0) \phi1_(i1)(q1) (d\phi1_(i2)/dx)(q2) (du/dx2)(q0, q1,
  //    q2)
  //
  //    = (d\phi1_(i0)/dx)(q0) \delta_(i1, q1) \delta_(i2, q2) (du/dx0)(q0, q1,
  //    q2)
  //    + \delta_(i0, q0) (d\phi1_(i1)/dx)(q1) \delta_(i2, q2) (du/dx1)(q0, q1,
  //    q2)
  //    + \delta_(i0, q0) \delta_(i1, q1) (d\phi1_(i2)/dx)(q2) (du/dx2)(q0, q1,
  //    q2)
  //
  //    = d\phi1_(i0)/dx(q0) (du/dx0)(q0, i1, i2) + (d\phi1_(i1)/dx)(q1)
  //    (du/dx1)(i0, q1, i2) + (d\phi1_(i2)/dx)(q2) (du/dx2)(i0, i1, q2)
  __syncthreads();
  T yd = 0;
  for (int idx = 0; idx < nq; ++idx)
  {
    yd += dphi1[idx * nq1 + tx] * scratch1[ijk<square_nq, nq, 1>(idx, ty, tz)];
    yd += dphi1[idx * nq1 + ty] * scratch2[ijk<square_nq, nq, 1>(tx, idx, tz)];
    yd += dphi1[idx * nq1 + tz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, idx)];
  }

  // Interpolate quadrature points to dofs
  if (identity != 1)
  {
    // Note that v1 = Pi v0 (v1, v0 are dofs), i.e. v1_(i) = Pi_(i, j) v0_(j),
    // where \Pi_(ij) = \Phi0_(j)(xi) and \Phi0_(i)(xj) is \Phi0_(i)
    // evaluated at node the node of \Phi1_(j). Therefore
    //
    //  vh(x) = \Phi0_(j)(x) v0_j
    //        = \Phi1_(j)(x) \Pi_(ji) v0_i
    //        = \Phi1_(j)(x) \Phi0_(i)(xj) v0_i
    //        = \Phi1_(j)(x) \phi0_(i0)(xj_0) \phi0_(i1)(xj_1j) \phi0_(i2)(xj_2)
    //        v0_(i0, i1, i2)
    //
    // hence
    //
    //  \Phi0(i)(x) = \Phi1_(j)(x) \phi0_(i0)(x_(j0)) \phi0_(i1)(x_(j1))
    //  \phi0_(i2)(x_(j2))
    //
    // and
    //
    //  \Phi0_(i0, i1, i2)(x0, x1, x2) = \phi1_(j0)(x0) \phi1_(j1)(x1)
    //  \phi1_(j2)(x2)
    //         \phi0_(i0)(x_(j0)) \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // Hence:
    //
    //  (d\Phi0_(i0, i1, i2)/dx0)(x0, x1, x2)
    //     = (d\phi1_(j0)/dx)(x0) \phi1_(j1)(x1) \phi1_(j2)(x2)
    //     \phi0_(i0)(x_(j0))
    //          \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // At quadrature points,
    //
    //  (d\Phi0_(i0, i1, i2)/dx0)(q0, q1, q2)
    //     = (d\phi1_(j0)/dx)(q0) \phi1_(j1)(q1) \phi1_(j2)(q2)
    //     \phi0_(i0)(x_(j0))
    //     \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2)) = (d\phi1_(j0)/dx)(q0)
    //     \phi0_(i0)(x_(j0))
    //     \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // We want to compute
    //
    // r0_(i0, i1, i2) = (d\Phi0_(i0, i1, i2)/dx0)(q0, q1, q2) (du/dx0)(q0, q1,
    // q2)
    //                 + (d\Phi0_(i0, i1, i2)/dx1)(q0, q1, q2) (du/dx1)(q0, q1,
    //                 q2)
    //                 + (d\Phi0_(i0, i1, i2)/dx2)(q0, q1, q2) (du/dx2)(q0, q1,
    //                 q2)
    //
    //                 = (d\phi1_(q0)/dx)(q0) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx0)(q0, q1, q2)
    //                 + (d\phi1_(q1)/dx)(q1) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx1)(q0, q1, q2)
    //                 + (d\phi1_(q2)/dx)(q2) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx2)(q0, q1, q2)
    //
    //                 = \phi0_(i0)(q0) \phi0_(i1)(q1) \phi0_(i2)(q2)
    //                 [(d\phi1_(q0)/dx)(q0) du/dx0_(q0, q1, q2)
    //                     + (d\phi1_(q1)/dx)(q1) du/dx1_(q0, q1, q2) +
    //                     (d\phi1_(q2)/dx)(q2) du/dx2_(q0, q1, q2)]
    //
    // Have already computed
    //
    // r1_(q0, q1, q2)  = d\phi1_(q0)/dx(q0) du/dx0_(q0, i1, i2) +
    // d\phi1_(q1)/dx(q1) du/dx1_(i0, q1, i2) + d\phi1_(q2)/dx(q2) du/dx2_(i0,
    // i1, q2)
    //
    // So we compute:
    //
    //  r0_(i0, i1, i2) = \phi0_(i0)(q0) \phi0_(i1)(q1) \phi0_(i2)(q2)
    //  [(d\phi1_(q0)/dx)(q0) du/dx0_(q0, q1, q2) r1_(q0, q1, q2)

    __syncthreads();
    scratch1[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;

    __syncthreads();
    yd = 0;
    if (tx < nd)
    {
      // tmp0(i0, q1, q2) += phi0_(i0)(q0) * r1(q0, q1, q2)
      for (int ix = 0; ix < nq; ++ix)
        yd += phi0[ix * nd + tx] * scratch1[ijk<square_nq, nq, 1>(ix, ty, tz)];
    }

    scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;
    __syncthreads();

    yd = 0;
    if (ty < nd)
    {
      // tmp1(i0, i1, q2) += phi0_(i1)(q1) * tmp0(i0, q1, q2)
      for (int iy = 0; iy < nq; ++iy)
        yd += phi0[iy * nd + ty] * scratch2[ijk<square_nq, nq, 1>(tx, iy, tz)];
    }

    scratch3[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;
    __syncthreads();

    yd = 0;
    if (tz < nd)
    {
      // b(i0, i1, i2) += phi0_(i2)(q2) * tmp1(i0, i1, q2)
      for (int iz = 0; iz < nq; ++iz)
        yd += phi0[iz * nd + tz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, iz)];
    }
  } // end of interpolation

  // Write back to global memory
  if (dof != -1)
    atomicAdd(&b[dof], yd);
}

} // namespace benchdolfinx

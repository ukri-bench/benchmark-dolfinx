// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>

namespace benchdolfinx
{
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
/// @param phi0 Array of size (nq, ndofs) with the interpolation basis
/// functions in 1D. u1_i = phi0_(ij) u_j, where u are the dofs
/// associated with the element (degree P), and u1 are the dofs for the
/// finite elment (degree >= P) that u is interpolated into.
/// @param dphi1 Array of size (nq, nq) with the 1D basis function
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
void stiffness_operator(const T* __restrict__ u,
                        const T* __restrict__ entity_constants,
                        T* __restrict__ b, const T* __restrict__ G_entity,
                        const T* __restrict__ phi0, const T* __restrict__ dphi1,
                        const std::int32_t* __restrict__ entity_dofmap,
                        const int* __restrict__ entities, int n_entities,
                        const std::int8_t* __restrict__ bc_marker,
                        bool identity)
{
  constexpr int nd = (P + 1);
  constexpr int cube_nd = nd * nd * nd;
  constexpr int nq = Q;
  constexpr int cube_nq = nq * nq * nq;

  // FIXME: qmode0 only
  static_assert(Q == P + 1);

  T local_dofs[nd][nd][nd];
  for (int c = 0; c < n_entities; ++c)
  {
    int entity_index = entities[c];
    // Copy input dofs for this cell
    const std::int32_t* dofs = entity_dofmap + entity_index * cube_nd;
    for (int ix = 0; ix < nd; ++ix)
      for (int iy = 0; iy < nd; ++iy)
        for (int iz = 0; iz < nd; ++iz)
        {
          std::int32_t dof = dofs[ix * nd * nd + iy * nd + iz];
          if (bc_marker[dof])
          {
            b[dof] = u[dof];
            local_dofs[ix][iy][iz] = 0;
          }
          else
            local_dofs[ix][iy][iz] = u[dof];
        }

    T coeff = entity_constants[c];

    T scratch1[nq][nq][nq];
    T scratch2[nq][nq][nq];
    T scratch3[nq][nq][nq];

    // (du/dx0)_(q0, q1, q2) = (d\phi1_(i)/dx)(q0) u_(i, q1, q2)
    for (int ix = 0; ix < nq; ++ix)
      for (int iy = 0; iy < nq; ++iy)
        for (int iz = 0; iz < nq; ++iz)
        {
          const T* G = G_entity + c * cube_nq * 6 + ix * nq * nq + iy * nq + iz;
          const T G0 = G[0];
          const T G1 = G[cube_nq];
          const T G2 = G[2 * cube_nq];
          const T G3 = G[3 * cube_nq];
          const T G4 = G[4 * cube_nq];
          const T G5 = G[5 * cube_nq];

          T val_x = 0;
          T val_y = 0;
          T val_z = 0;
          for (int i = 0; i < nq; ++i)
          {
            val_x += dphi1[ix * nq + i] * local_dofs[i][iy][iz];
            val_y += dphi1[iy * nq + i] * local_dofs[ix][i][iz];
            val_z += dphi1[iz * nq + i] * local_dofs[ix][iy][i];
          }
          scratch1[ix][iy][iz] = coeff * (G0 * val_x + G1 * val_y + G2 * val_z);
          scratch2[ix][iy][iz] = coeff * (G1 * val_x + G3 * val_y + G4 * val_z);
          scratch3[ix][iy][iz] = coeff * (G2 * val_x + G4 * val_y + G5 * val_z);
        }

    // (du/dx0)_(q0, q1, q2) = (d\phi1_(i)/dx)(q0) u_(i, q1, q2)
    for (int ix = 0; ix < nd; ++ix)
      for (int iy = 0; iy < nd; ++iy)
        for (int iz = 0; iz < nd; ++iz)
        {
          T yd = 0;
          for (int i = 0; i < nq; ++i)
          {
            yd += dphi1[i * nq + ix] * scratch1[i][iy][iz];
            yd += dphi1[i * nq + iy] * scratch2[ix][i][iz];
            yd += dphi1[i * nq + iz] * scratch3[ix][iy][i];
          }
          std::int32_t dof = dofs[ix * nd * nd + iy * nd + iz];
          if (!bc_marker[dof])
            b[dof] += yd;
        }
  }
}

} // namespace benchdolfinx

// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace benchdolfinx
{
template <typename T, int Q>
void geometry_computation(const T* xgeom, T* G_entity,
                          const std::int32_t* geometry_dofmap, const T* _dphi,
                          const T* weights, const int* entities, int n_entities)
{
  for (int c = 0; c < n_entities; ++c)
  {
    // Cell index
    int cell = entities[c];

    // Number of quadrature points (must match arrays in weights and
    // dphi)
    constexpr int nq = Q * Q * Q;

    // Number of coordinate dofs
    constexpr int ncdofs = 8;

    // Geometric dimension
    constexpr int gdim = 3;

    // coord_dofs has shape [ncdofs, gdim]
    T _coord_dofs[ncdofs * gdim];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 3; ++j)
      {
        _coord_dofs[i * 3 + j]
            = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
      }

    // Jacobian
    T J[3][3];
    auto coord_dofs = [&_coord_dofs](int i, int j) -> T&
    { return _coord_dofs[i * gdim + j]; };

    for (int iq = 0; iq < nq; ++iq)
    // For each quadrature point
    {
      // dphi has shape [gdim, ncdofs]
      auto dphi = [&_dphi, iq](int i, int j) -> const T
      { return _dphi[(i * nq + iq) * ncdofs + j]; };
      for (std::size_t i = 0; i < gdim; i++)
      {
        for (std::size_t j = 0; j < gdim; j++)
        {
          J[i][j] = 0.0;
          for (std::size_t k = 0; k < ncdofs; k++)
            J[i][j] += coord_dofs(k, i) * dphi(j, k);
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

      int offset = (c * nq * 6 + iq);
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
}

} // namespace benchdolfinx

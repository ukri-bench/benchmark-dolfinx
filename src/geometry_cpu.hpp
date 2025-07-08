// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include <cstdint>

#pragma once

namespace benchdolfinx
{
template <typename T, int Q>
void geometry_computation(const T* xgeom, T* G_entity,
                          const std::int32_t* geometry_dofmap, const T* _dphi,
                          const T* weights, const int* entities,
                          int n_entities);
}

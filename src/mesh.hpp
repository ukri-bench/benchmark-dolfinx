// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

namespace benchdolfinx
{
/// @brief Create a new mesh with an extra boundary layer, such that all
/// cells on other processes which share a vertex with this process are
/// ghosted.
/// @param mesh Input mesh
/// @param coord_element A coordinate element for the new mesh. This may
/// be tensor product ordering.
template <std::floating_point T>
dolfinx::mesh::Mesh<T>
ghost_layer_mesh(dolfinx::mesh::Mesh<T>& mesh,
                 dolfinx::fem::CoordinateElement<T> coord_element);

template <typename T>
std::array<std::vector<std::int32_t>, 2>
compute_boundary_cells(const dolfinx::fem::FunctionSpace<T>& V);

} // namespace benchdolfinx

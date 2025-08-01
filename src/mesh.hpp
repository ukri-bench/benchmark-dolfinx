// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

namespace benchdolfinx
{
/// @brief Compute shape of a cube mesh which will support a functionspace with
/// approximately the desired number of dofs.
/// @param ndofs The desired number of dofs
/// @param degree The polynomial degree of the functionspace
/// @param mpi_size Number of ranks
/// @return shape of the cube mesh (nx, ny, nz)
std::array<std::int64_t, 3> compute_mesh_size(std::int64_t ndofs, int degree,
                                              int mpi_size);

/// @brief Create a cube mesh of size n[0] x n[1] x n[2] with an appropriate
/// ghost layer, so that local and boundary cells can be treated in separate
/// steps in a solver.
/// @tparam T Scalar type
/// @param comm MPI Communicator
/// @param n Number of cells in each direction
/// @param geom_perturb_fact Random perturbation to the geometry by this factor
/// @return A mesh
template <typename T>
dolfinx::mesh::Mesh<T> create_mesh(MPI_Comm comm, std::array<std::int64_t, 3> n,
                                   T geom_perturb_fact);

/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
/// @param V FunctionSpace of the degrees-of-freedom
/// @returns An array of two lists: (local cells, boundary cells)
template <typename T>
std::array<std::vector<std::int32_t>, 2>
compute_boundary_cells(const dolfinx::fem::FunctionSpace<T>& V);

} // namespace benchdolfinx

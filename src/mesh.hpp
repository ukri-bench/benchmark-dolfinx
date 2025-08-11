// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <concepts>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

namespace dolfinx
{
namespace fem
{
class DofMap;
}
namespace mesh
{
class Topology;
}
} // namespace dolfinx

namespace benchdolfinx
{
/// @brief Compute number of cells in each direction of a cube mesh with
/// hexahedral cells such that the number of degrees-of-freedom is close
/// to a prescribed target number of degrees-of-freedom.
///
/// Computation of the number of cells is for scalar continuous Lagrange
/// finite elements.
///
/// @todo Why is `mpi_size` required?
///
/// @param ndofs Target number of degrees-of-freedom.
/// @param degree Polynomial degree of the element.
/// @param mpi_size Number of MPI ranks.
/// @return Number of cells in each axis direction.
std::array<std::int64_t, 3> compute_mesh_size(std::int64_t ndofs, int degree,
                                              int mpi_size);

/// @brief Create a cube mesh with `n[0] x n[1] x n[2]` cells in each
/// direction.
///
/// The mesh is constructed with wan appropriate ghost layer such so
/// that local (interior) and boundary cells can be treated in separate
/// steps in a solver.
///
/// @todo Text of the ghost layer is vague. Need to me make more precise.
///
/// @tparam T Geometry scalar type.
/// @param comm MPI Communicator
/// @param n Number of cells in each direction
/// @param geom_perturb_fact Random perturbation to the geometry by this factor
/// @return A mesh
template <std::floating_point T>
dolfinx::mesh::Mesh<T> create_mesh(MPI_Comm comm, std::array<std::int64_t, 3> n,
                                   T geom_perturb_fact);

/// @brief Compute cell index partitions based on sharing.
///
/// The groups are:
/// 1. Cells which are "local", i.e. the dofs on these cells are not
///    shared with any other process.
/// 2. Cells which share degrees-of-freedom with other processes.
///
/// @param V Function space of the finite element space
/// @returns Lists of cells: (0) local cells and (1) boundary cells.
std::array<std::vector<std::int32_t>, 2>
compute_boundary_cells(const dolfinx::mesh::Topology& topology,
                       const dolfinx::fem::DofMap& dofmap);

} // namespace benchdolfinx

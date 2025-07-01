// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "mesh.hpp"
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

namespace
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
                 dolfinx::fem::CoordinateElement<T> coord_element)
{
  constexpr int tdim = 3;
  constexpr int gdim = 3;
  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations
      = mesh.topology()->index_map(0)->index_to_dest_ranks();

  // Map from any local cells to processes where they should be ghosted
  std::map<int, std::vector<int>> cell_to_dests;
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  std::vector<int> cdests;
  for (std::size_t c = 0; c < ncells; ++c)
  {
    cdests.clear();
    for (auto v : c_to_v->links(c))
    {
      auto vdest = vertex_destinations.links(v);
      for (int dest : vdest)
        cdests.push_back(dest);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    if (!cdests.empty())
      cell_to_dests[c] = cdests;
  }

  spdlog::info("cell_to_dests= {}, ncells = {}", cell_to_dests.size(), ncells);

  auto partitioner
      = [cell_to_dests,
         ncells](MPI_Comm comm, int nparts,
                 const std::vector<dolfinx::mesh::CellType>& cell_types,
                 const std::vector<std::span<const std::int64_t>>& cells)
  {
    int rank = dolfinx::MPI::rank(comm);
    std::vector<std::int32_t> dests;
    std::vector<int> offsets = {0};
    for (std::size_t c = 0; c < ncells; ++c)
    {
      dests.push_back(rank);
      if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
        dests.insert(dests.end(), it->second.begin(), it->second.end());

      // Ghost to other processes
      offsets.push_back(dests.size());
    }
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests),
                                                       std::move(offsets));
  };

  std::array<std::size_t, 2> xshape = {num_vertices, gdim};
  std::span<T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  auto dofmap = mesh.geometry().dofmap();
  auto imap = mesh.geometry().index_map();
  std::vector<std::int32_t> permuted_dofmap;
  std::vector<int> perm = basix::tp_dof_ordering(
      basix::element::family::P,
      dolfinx::mesh::cell_type_to_basix_type(coord_element.cell_shape()),
      coord_element.degree(), coord_element.variant(),
      basix::element::dpc_variant::unset, false);
  for (std::size_t c = 0; c < dofmap.extent(0); ++c)
  {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (std::size_t i = 0; i < dofmap.extent(1); ++i)
      permuted_dofmap.push_back(cell_dofs(perm[i]));
  }
  std::vector<std::int64_t> permuted_dofmap_global(permuted_dofmap.size());
  imap->local_to_global(permuted_dofmap, permuted_dofmap_global);

  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
      coord_element, mesh.comm(), x, xshape, partitioner);

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());

  return new_mesh;
}
} // namespace

/// @brief TODO
/// @param ndofs
/// @param degree
/// @param mpi_size
/// @return
std::array<std::int64_t, 3>
benchdolfinx::compute_mesh_size(std::int64_t ndofs, int degree, int mpi_size)
{
  double nx_approx = (std::pow(ndofs * mpi_size, 1.0 / 3.0) - 1) / degree;
  std::int64_t n0 = static_cast<std::int64_t>(nx_approx);
  std::array<std::int64_t, 3> nx = {n0, n0, n0};

  // Try to improve fit to ndofs +/- 5 in each direction
  if (n0 > 5)
  {
    std::int64_t best_misfit
        = (n0 * degree + 1) * (n0 * degree + 1) * (n0 * degree + 1)
          - ndofs * mpi_size;
    best_misfit = std::abs(best_misfit);
    for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
    {
      for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
      {
        for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
        {
          std::int64_t misfit
              = (nx0 * degree + 1) * (ny0 * degree + 1) * (nz0 * degree + 1)
                - ndofs * mpi_size;
          if (std::abs(misfit) < best_misfit)
          {
            best_misfit = std::abs(misfit);
            nx = {nx0, ny0, nz0};
          }
        }
      }
    }
  }

  return nx;
}

/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
template <typename T>
std::array<std::vector<std::int32_t>, 2>
benchdolfinx::compute_boundary_cells(const dolfinx::fem::FunctionSpace<T>& V)
{
  auto mesh = V.mesh();
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  std::int32_t ncells_local = topology->index_map(tdim)->size_local();
  std::int32_t ncells_ghost = topology->index_map(tdim)->num_ghosts();
  std::int32_t ndofs_local = V.dofmap()->index_map->size_local();

  std::vector<std::uint8_t> cell_mark(ncells_local + ncells_ghost, 0);
  for (int i = 0; i < ncells_local; ++i)
  {
    auto cell_dofs = V.dofmap()->cell_dofs(i);
    for (auto dof : cell_dofs)
      if (dof >= ndofs_local)
        cell_mark[i] = 1;
  }
  for (int i = ncells_local; i < ncells_local + ncells_ghost; ++i)
    cell_mark[i] = 1;

  std::vector<std::int32_t> local_cells;
  std::vector<std::int32_t> boundary_cells;
  for (std::size_t i = 0; i < cell_mark.size(); ++i)
  {
    if (cell_mark[i])
      boundary_cells.push_back(i);
    else
      local_cells.push_back(i);
  }

  spdlog::debug("lcells:{}, bcells:{}", local_cells.size(),
                boundary_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}

/// @brief Create a cube mesh of size n[0] x n[1] x n[2] with an appropriate
/// ghost layer, so that local and boundary cells can be treated in separate
/// steps in a solver.
/// @tparam T Scalar type
/// @param comm MPI Communicator
/// @param n Number of cells in each direction
/// @param geom_perturb_fact Random perturbation to the geometry by this factor
/// @return A mesh
template <typename T>
dolfinx::mesh::Mesh<T> benchdolfinx::create_mesh(MPI_Comm comm,
                                                 std::array<std::int64_t, 3> n,
                                                 T geom_perturb_fact)
{
  dolfinx::mesh::Mesh<T> mesh0 = dolfinx::mesh::create_box<T>(
      comm, {{{0, 0, 0}, {1, 1, 1}}}, {n[0], n[1], n[2]},
      dolfinx::mesh::CellType::hexahedron);

  if (geom_perturb_fact != 0.0)
  {
    double perturb_x = geom_perturb_fact * 1 / n[0];
    std::span geom_x = mesh0.geometry().x();
    std::mt19937 generator(42);
    std::uniform_real_distribution<T> distribution(-perturb_x, perturb_x);
    for (std::size_t i = 0; i < geom_x.size(); i += 3)
      geom_x[i] += distribution(generator);
  }

  // Degree 1 coordinate element
  auto element
      = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
          basix::element::family::P, basix::cell::type::hexahedron, 1,
          basix::element::lagrange_variant::gll_warped,
          basix::element::dpc_variant::unset, false));
  dolfinx::fem::CoordinateElement<T> celement(element);

  return ghost_layer_mesh(mesh0, celement);
}

// Explicit instantiation for double and float
template dolfinx::mesh::Mesh<double>
benchdolfinx::create_mesh<double>(MPI_Comm comm, std::array<std::int64_t, 3> n,
                                  double geom_perturb_fact);
template dolfinx::mesh::Mesh<float>
benchdolfinx::create_mesh<float>(MPI_Comm comm, std::array<std::int64_t, 3> n,
                                 float geom_perturb_fact);

template std::array<std::vector<std::int32_t>, 2>
benchdolfinx::compute_boundary_cells(
    const dolfinx::fem::FunctionSpace<double>& V);
template std::array<std::vector<std::int32_t>, 2>
benchdolfinx::compute_boundary_cells(
    const dolfinx::fem::FunctionSpace<float>& V);

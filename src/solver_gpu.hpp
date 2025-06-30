// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

namespace benchdolfinx
{

template <typename T>
void solver_gpu(const dolfinx::mesh::Mesh<T>& mesh)
{
  // Finite element for higher-order discretisation
  auto element = basix::create_tp_element<T>(
      basix::element::family::P, basix::cell::type::hexahedron, order,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  const std::int32_t num_cells_all
      = mesh->topology()->index_map(tdim)->size_local()
        + mesh->topology()->index_map(tdim)->num_ghosts();

  // TODO: get Constant from Form
  // thrust::device_vector<T> constants_d(num_cells_all, kappa->value[0]);
  thrust::device_vector<T> constants_d(num_cells_all, 10);
}

} // namespace benchdolfinx

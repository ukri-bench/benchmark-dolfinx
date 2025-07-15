// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>

namespace benchdolfinx
{
template <typename T>
void laplace_action(const dolfinx::fem::Form<T>& a,
                    const dolfinx::fem::Form<T>& L,
                    const dolfinx::fem::DirichletBC<T>& bc, int degree,
                    int qmode, T kappa, int nreps, bool use_gauss);
} // namespace benchdolfinx

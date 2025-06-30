// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#if defined(USE_CUDA) || defined(USE_HIP)

#include "geometry_gpu.hpp"
#include "laplacian_gpu.hpp"
#include "mesh.hpp"
#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace benchdolfinx
{
template <typename T>
void laplace_action(const dolfinx::fem::Form<T>& a,
                    const dolfinx::fem::Form<T>& L,
                    const dolfinx::fem::DirichletBC<T>& bc, int degree,
                    int qmode, T kappa, int nreps, bool use_gauss);
} // namespace benchdolfinx

#endif

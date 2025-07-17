// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>

namespace benchdolfinx
{
struct BenchmarkResults
{
  double mat_free_time;
  double enorm;
  double unorm;
  double ynorm;
  double znorm;
};

using BenchmarkResults = struct BenchmarkResults;

/// @brief Compute action of Laplacian on GPU or CPU
template <typename T>
BenchmarkResults
laplace_action(const dolfinx::fem::Form<T>& a, const dolfinx::fem::Form<T>& L,
               const dolfinx::fem::DirichletBC<T>& bc, int degree, int qmode,
               T kappa, int nreps, bool use_gauss, bool matrix_comparison);
} // namespace benchdolfinx

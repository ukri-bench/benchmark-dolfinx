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

#if defined(USE_CUDA) || defined(USE_HIP)
/// @brief Compute action of Laplacian on GPU
template <typename T>
BenchmarkResults laplace_action_gpu(const dolfinx::fem::Form<T>& a,
                                    const dolfinx::fem::Form<T>& L,
                                    const dolfinx::fem::DirichletBC<T>& bc,
                                    int degree, int qmode, T kappa, int nreps,
                                    bool use_gauss, bool matrix_comparison,
                                    bool use_cg);
#endif

/// @brief Compute action of Laplacian on CPU
template <typename T>
BenchmarkResults laplace_action_cpu(const dolfinx::fem::Form<T>& a,
                                    const dolfinx::fem::Form<T>& L,
                                    const dolfinx::fem::DirichletBC<T>& bc,
                                    int degree, int qmode, T kappa, int nreps,
                                    bool use_gauss, bool matrix_comparison,
                                    bool use_cg);
} // namespace benchdolfinx

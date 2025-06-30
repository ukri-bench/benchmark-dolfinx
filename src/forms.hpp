// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <map>
#include <memory>
#include <string>

namespace benchdolfinx
{
dolfinx::fem::Form<double> create_laplacian_form2(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Constant<double>>>&
        coeffs,
    int qmode, bool use_gauss, int degree);

dolfinx::fem::Form<double> create_laplacian_form1(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    const std::map<
        std::string,
        std::shared_ptr<const dolfinx::fem::Function<double, double>>>& coeffs,
    int qmode, bool use_gauss, int degree);
} // namespace benchdolfinx

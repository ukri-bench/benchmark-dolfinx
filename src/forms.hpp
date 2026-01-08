// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <map>
#include <memory>
#include <string>

namespace benchdolfinx
{
/// @brief Create bilinear form (LHS)
/// @param V FunctionSpace on which Form is defined
/// @param coeffs Coefficients of the form
/// @param qmode Quadrature mode (0=colocated nq=p+1, 1=interpolated nq=p+2)
/// @param use_gauss Use Gauss quadrature
/// @param degree Polynomial degree
/// @tparam T Scalar type (float or double)
template <typename T>
dolfinx::fem::Form<T> create_laplacian_form2(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<T>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Constant<T>>>& coeffs,
    int qmode, bool use_gauss, int degree);

/// @brief Create linear form (RHS)
/// @param V FunctionSpace on which Form is defined
/// @param coeffs Coefficients of the form
/// @param qmode Quadrature mode (0=colocated nq=p+1, 1=interpolated nq=p+2)
/// @param use_gauss Use Gauss quadrature
/// @param degree Polynomial degree
/// @tparam T Scalar type (float or double)
template <typename T>
dolfinx::fem::Form<T> create_laplacian_form1(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<T>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Function<T, T>>>& coeffs,
    int qmode, bool use_gauss, int degree);
} // namespace benchdolfinx

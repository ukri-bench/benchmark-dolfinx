// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "forms.hpp"
#include "poisson.h"
#include <array>
#include <dolfinx/fem/utils.h>
#include <vector>

using namespace dolfinx;

namespace
{
// qmode == 0, Gauss
static std::vector aforms_gl0{form_poisson_a_1_2_GL, form_poisson_a_2_3_GL,
                              form_poisson_a_3_4_GL, form_poisson_a_4_5_GL,
                              form_poisson_a_5_6_GL, form_poisson_a_6_7_GL,
                              form_poisson_a_7_8_GL};
static std::vector Lforms_gl0{form_poisson_L_1_2_GL, form_poisson_L_2_3_GL,
                              form_poisson_L_3_4_GL, form_poisson_L_4_5_GL,
                              form_poisson_L_5_6_GL, form_poisson_L_6_7_GL,
                              form_poisson_L_7_8_GL};

// qmode == 1, Gauss
static std::vector aforms_gl1{form_poisson_a_1_3_GL, form_poisson_a_2_4_GL,
                              form_poisson_a_3_5_GL, form_poisson_a_4_6_GL,
                              form_poisson_a_5_7_GL, form_poisson_a_6_8_GL,
                              form_poisson_a_7_9_GL};
static std::vector Lforms_gl1{form_poisson_L_1_3_GL, form_poisson_L_2_4_GL,
                              form_poisson_L_3_5_GL, form_poisson_L_4_6_GL,
                              form_poisson_L_5_7_GL, form_poisson_L_6_8_GL,
                              form_poisson_L_7_9_GL};

// qmode == 0, GLL
static std::vector aforms_gll0{form_poisson_a_1_2_GLL, form_poisson_a_2_3_GLL,
                               form_poisson_a_3_4_GLL, form_poisson_a_4_5_GLL,
                               form_poisson_a_5_6_GLL, form_poisson_a_6_7_GLL,
                               form_poisson_a_7_8_GLL};
static std::vector Lforms_gll0{form_poisson_L_1_2_GLL, form_poisson_L_2_3_GLL,
                               form_poisson_L_3_4_GLL, form_poisson_L_4_5_GLL,
                               form_poisson_L_5_6_GLL, form_poisson_L_6_7_GLL,
                               form_poisson_L_7_8_GLL};

// qmode == 1, GLL
static std::vector aforms_gll1{form_poisson_a_1_3_GLL, form_poisson_a_2_4_GLL,
                               form_poisson_a_3_5_GLL, form_poisson_a_4_6_GLL,
                               form_poisson_a_5_7_GLL, form_poisson_a_6_8_GLL,
                               form_poisson_a_7_9_GLL};
static std::vector Lforms_gll1{form_poisson_L_1_3_GLL, form_poisson_L_2_4_GLL,
                               form_poisson_L_3_5_GLL, form_poisson_L_4_6_GLL,
                               form_poisson_L_5_7_GLL, form_poisson_L_6_8_GLL,
                               form_poisson_L_7_9_GLL};
} // namespace

//----------------------------------------------------------------------------
dolfinx::fem::Form<double> benchdolfinx::create_laplacian_form2(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Constant<double>>>&
        coeffs,
    int qmode, bool use_gauss, int degree)
{
  ufcx_form* form = nullptr;
  if (qmode == 0)
  {
    if (use_gauss)
      form = aforms_gl0.at(degree - 1);
    else
      form = aforms_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = aforms_gl1.at(degree - 1);
    else
      form = aforms_gll1.at(degree - 1);
  }

  return fem::create_form<double>(*form, {V, V}, {}, coeffs, {}, {});
}
//----------------------------------------------------------------------------
dolfinx::fem::Form<double> benchdolfinx::create_laplacian_form1(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    const std::map<
        std::string,
        std::shared_ptr<const dolfinx::fem::Function<double, double>>>& coeffs,
    int qmode, bool use_gauss, int degree)
{
  ufcx_form* form = nullptr;
  if (qmode == 0)
  {
    if (use_gauss)
      form = Lforms_gl0.at(degree - 1);
    else
      form = Lforms_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = Lforms_gl1.at(degree - 1);
    else
      form = Lforms_gll1.at(degree - 1);
  }

  return fem::create_form<double>(*form, {V}, coeffs, {}, {}, {});
}
//----------------------------------------------------------------------------

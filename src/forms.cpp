// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "forms.hpp"
#include "poisson32.h"
#include "poisson64.h"
#include <array>
#include <dolfinx/fem/utils.h>
#include <vector>

using namespace dolfinx;

namespace
{
// qmode == 0, Gauss
static std::vector aforms32_gl0{
    form_poisson32_a_1_2_GL, form_poisson32_a_2_3_GL, form_poisson32_a_3_4_GL,
    form_poisson32_a_4_5_GL, form_poisson32_a_5_6_GL, form_poisson32_a_6_7_GL,
    form_poisson32_a_7_8_GL};
static std::vector Lforms32_gl0{
    form_poisson32_L_1_2_GL, form_poisson32_L_2_3_GL, form_poisson32_L_3_4_GL,
    form_poisson32_L_4_5_GL, form_poisson32_L_5_6_GL, form_poisson32_L_6_7_GL,
    form_poisson32_L_7_8_GL};

// qmode == 1, Gauss
static std::vector aforms32_gl1{
    form_poisson32_a_1_3_GL, form_poisson32_a_2_4_GL, form_poisson32_a_3_5_GL,
    form_poisson32_a_4_6_GL, form_poisson32_a_5_7_GL, form_poisson32_a_6_8_GL,
    form_poisson32_a_7_9_GL};
static std::vector Lforms32_gl1{
    form_poisson32_L_1_3_GL, form_poisson32_L_2_4_GL, form_poisson32_L_3_5_GL,
    form_poisson32_L_4_6_GL, form_poisson32_L_5_7_GL, form_poisson32_L_6_8_GL,
    form_poisson32_L_7_9_GL};

// qmode == 0, GLL
static std::vector aforms32_gll0{
    form_poisson32_a_1_2_GLL, form_poisson32_a_2_3_GLL,
    form_poisson32_a_3_4_GLL, form_poisson32_a_4_5_GLL,
    form_poisson32_a_5_6_GLL, form_poisson32_a_6_7_GLL,
    form_poisson32_a_7_8_GLL};
static std::vector Lforms32_gll0{
    form_poisson32_L_1_2_GLL, form_poisson32_L_2_3_GLL,
    form_poisson32_L_3_4_GLL, form_poisson32_L_4_5_GLL,
    form_poisson32_L_5_6_GLL, form_poisson32_L_6_7_GLL,
    form_poisson32_L_7_8_GLL};

// qmode == 1, GLL
static std::vector aforms32_gll1{
    form_poisson32_a_1_3_GLL, form_poisson32_a_2_4_GLL,
    form_poisson32_a_3_5_GLL, form_poisson32_a_4_6_GLL,
    form_poisson32_a_5_7_GLL, form_poisson32_a_6_8_GLL,
    form_poisson32_a_7_9_GLL};
static std::vector Lforms32_gll1{
    form_poisson32_L_1_3_GLL, form_poisson32_L_2_4_GLL,
    form_poisson32_L_3_5_GLL, form_poisson32_L_4_6_GLL,
    form_poisson32_L_5_7_GLL, form_poisson32_L_6_8_GLL,
    form_poisson32_L_7_9_GLL};

// qmode == 0, Gauss
static std::vector aforms64_gl0{
    form_poisson64_a_1_2_GL, form_poisson64_a_2_3_GL, form_poisson64_a_3_4_GL,
    form_poisson64_a_4_5_GL, form_poisson64_a_5_6_GL, form_poisson64_a_6_7_GL,
    form_poisson64_a_7_8_GL};
static std::vector Lforms64_gl0{
    form_poisson64_L_1_2_GL, form_poisson64_L_2_3_GL, form_poisson64_L_3_4_GL,
    form_poisson64_L_4_5_GL, form_poisson64_L_5_6_GL, form_poisson64_L_6_7_GL,
    form_poisson64_L_7_8_GL};

// qmode == 1, Gauss
static std::vector aforms64_gl1{
    form_poisson64_a_1_3_GL, form_poisson64_a_2_4_GL, form_poisson64_a_3_5_GL,
    form_poisson64_a_4_6_GL, form_poisson64_a_5_7_GL, form_poisson64_a_6_8_GL,
    form_poisson64_a_7_9_GL};
static std::vector Lforms64_gl1{
    form_poisson64_L_1_3_GL, form_poisson64_L_2_4_GL, form_poisson64_L_3_5_GL,
    form_poisson64_L_4_6_GL, form_poisson64_L_5_7_GL, form_poisson64_L_6_8_GL,
    form_poisson64_L_7_9_GL};

// qmode == 0, GLL
static std::vector aforms64_gll0{
    form_poisson64_a_1_2_GLL, form_poisson64_a_2_3_GLL,
    form_poisson64_a_3_4_GLL, form_poisson64_a_4_5_GLL,
    form_poisson64_a_5_6_GLL, form_poisson64_a_6_7_GLL,
    form_poisson64_a_7_8_GLL};
static std::vector Lforms64_gll0{
    form_poisson64_L_1_2_GLL, form_poisson64_L_2_3_GLL,
    form_poisson64_L_3_4_GLL, form_poisson64_L_4_5_GLL,
    form_poisson64_L_5_6_GLL, form_poisson64_L_6_7_GLL,
    form_poisson64_L_7_8_GLL};

// qmode == 1, GLL
static std::vector aforms64_gll1{
    form_poisson64_a_1_3_GLL, form_poisson64_a_2_4_GLL,
    form_poisson64_a_3_5_GLL, form_poisson64_a_4_6_GLL,
    form_poisson64_a_5_7_GLL, form_poisson64_a_6_8_GLL,
    form_poisson64_a_7_9_GLL};
static std::vector Lforms64_gll1{
    form_poisson64_L_1_3_GLL, form_poisson64_L_2_4_GLL,
    form_poisson64_L_3_5_GLL, form_poisson64_L_4_6_GLL,
    form_poisson64_L_5_7_GLL, form_poisson64_L_6_8_GLL,
    form_poisson64_L_7_9_GLL};
} // namespace

//----------------------------------------------------------------------------
template <>
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
      form = aforms64_gl0.at(degree - 1);
    else
      form = aforms64_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = aforms64_gl1.at(degree - 1);
    else
      form = aforms64_gll1.at(degree - 1);
  }

  return fem::create_form<double>(*form, {V, V}, {}, coeffs, {}, {});
}
//----------------------------------------------------------------------------
template <>
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
      form = Lforms64_gl0.at(degree - 1);
    else
      form = Lforms64_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = Lforms64_gl1.at(degree - 1);
    else
      form = Lforms64_gll1.at(degree - 1);
  }

  return fem::create_form<double>(*form, {V}, coeffs, {}, {}, {});
}
//----------------------------------------------------------------------------
template <>
dolfinx::fem::Form<float> benchdolfinx::create_laplacian_form2(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<float>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Constant<float>>>&
        coeffs,
    int qmode, bool use_gauss, int degree)
{
  ufcx_form* form = nullptr;
  if (qmode == 0)
  {
    if (use_gauss)
      form = aforms32_gl0.at(degree - 1);
    else
      form = aforms32_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = aforms32_gl1.at(degree - 1);
    else
      form = aforms32_gll1.at(degree - 1);
  }

  return fem::create_form<float>(*form, {V, V}, {}, coeffs, {}, {});
}
//----------------------------------------------------------------------------
template <>
dolfinx::fem::Form<float> benchdolfinx::create_laplacian_form1(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<float>> V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Function<float, float>>>&
        coeffs,
    int qmode, bool use_gauss, int degree)
{
  ufcx_form* form = nullptr;
  if (qmode == 0)
  {
    if (use_gauss)
      form = Lforms32_gl0.at(degree - 1);
    else
      form = Lforms32_gll0.at(degree - 1);
  }
  else
  {
    if (use_gauss)
      form = Lforms32_gl1.at(degree - 1);
    else
      form = Lforms32_gll1.at(degree - 1);
  }

  return fem::create_form<float>(*form, {V}, coeffs, {}, {}, {});
}
//----------------------------------------------------------------------------

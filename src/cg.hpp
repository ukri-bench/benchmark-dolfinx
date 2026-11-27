// Copyright(C) 2025-2026 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <dolfinx/common/log.h>
#include <dolfinx/la/Vector.h>
#include <type_traits>
#include <vector>

#if defined(USE_CUDA) || defined(USE_HIP)
#include "vector.hpp"
#endif

namespace detail
{
/// True if `Vector` stores its data on the host (`std::vector`), false
/// if on a GPU device.
template <typename Vector>
constexpr bool is_host_vector
    = std::is_same_v<typename Vector::container_type,
                     std::vector<typename Vector::value_type>>;

/// @brief Compute r = alpha * x + y on the owned entries, for a host
/// or device vector.
template <typename S, typename Vector>
void axpy(Vector& r, S alpha, const Vector& x, const Vector& y)
{
  using T = typename Vector::value_type;
  if constexpr (is_host_vector<Vector>)
  {
    std::transform(
        x.array().begin(), x.array().begin() + x.index_map()->size_local(),
        y.array().begin(), r.array().begin(),
        [alpha](const T& vx, const T& vy) { return vx * alpha + vy; });
  }
  else
  {
#if defined(USE_CUDA) || defined(USE_HIP)
    benchdolfinx::axpy(r, alpha, x, y);
#endif
  }
}

/// @brief Compute the inner product of two host or device vectors.
///
/// The two vectors must have the same parallel layout.
///
/// @note Collective MPI operation
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename Vector>
auto inner_product(const Vector& a, const Vector& b)
{
  if constexpr (is_host_vector<Vector>)
    return dolfinx::la::inner_product(a, b);
  else
  {
#if defined(USE_CUDA) || defined(USE_HIP)
    return benchdolfinx::inner_product(a, b);
#endif
  }
}
} // namespace detail

/// Solve Ax = b using unpreconditioned CG
/// @param A Operator (with apply method) computes y=A.x. `apply` must
/// update the ghost entries of x before using them.
/// @param x Vector solution
/// @param b Vector RHS
/// @param max_iter Maximum number of iterations
/// @param rtol Tolerance (default 0.0 will enforce max_iter iterations)
template <typename Operator, typename Vector, typename S>
int cg_solve(Operator& A, Vector& x, const Vector& b, int max_iter,
             S rtol = 0.0)
{
  using T = typename Vector::value_type;
  static_assert(std::is_same<S, T>(), "Type mismatch between rtol and Vector");
  static_assert(std::is_same<typename Operator::value_type, T>(),
                "Type mismatch between Vector and Operator");

  T xnorm = detail::inner_product(x, x);
  spdlog::info("CG: xnorm = {}", xnorm);
  T bnorm = detail::inner_product(b, b);
  spdlog::info("CG: bnorm = {}", bnorm);

  Vector r(x.index_map(), x.bs());
  Vector y(x.index_map(), x.bs());
  Vector p(x.index_map(), x.bs());

  // Compute initial residual r0 = b - Ax0
  A.apply(x, y);
  detail::axpy(r, -1, y, b);
  detail::axpy(p, 0.0, r, r);

  T rnorm0 = detail::inner_product(p, r);
  T rnorm = rnorm0;

  spdlog::info("CG: rnorm0 = {}", rnorm0);

  // Iterations of CG
  const T rtol2 = rtol * rtol;

  int k = 0;
  while (k < max_iter)
  {
    ++k;

    // MatVec: y = A.p. The operator updates the ghost entries of p
    // (forward scatter) itself.
    A.apply(p, y);

    // Calculate alpha = r.r/p.y
    const T alpha = rnorm / detail::inner_product(p, y);

    // Update x and r
    // Update x (x <- x + alpha*p)
    detail::axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    detail::axpy(r, -alpha, y, r);

    T rnorm_new = detail::inner_product(r, r);

    const T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    SPDLOG_DEBUG("Iteration {}, residual {}, alpha={}, beta={}", k,
                 std::sqrt(rnorm), alpha, beta);

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + M^-1(r))
    detail::axpy(p, beta, p, r);
  }
  return k;
}

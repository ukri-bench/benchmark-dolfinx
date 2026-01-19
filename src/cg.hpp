
#include <dolfinx/common/MPI.h>

#if defined(USE_CUDA) || defined(USE_HIP)
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <type_traits>
#endif

namespace detail
{
template <typename S, typename Vector>
void axpy(Vector& r, S alpha, const Vector& x, const Vector& y)
{
  using T = typename Vector::value_type;
#if defined(USE_CUDA) || defined(USE_HIP)
  thrust::transform(thrust::device, x.array().begin(),
                    x.array().begin() + x.index_map()->size_local(),
                    y.array().begin(), r.array().begin(),
                    [alpha] __host__ __device__(const T& vx, const T& vy)
                    { return vx * alpha + vy; });
#else
  std::transform(x.array().begin(),
                 x.array().begin() + x.index_map()->size_local(),
                 y.array().begin(), r.array().begin(),
                 [alpha](const T& vx, const T& vy) { return vx * alpha + vy; });
#endif
}

/// Compute the inner product of two vectors. The two vectors must have
/// the same parallel layout
/// @note Collective MPI operation
/// @param a A vector
/// @param b A vector
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename Vector>
auto inner_product(const Vector& a, const Vector& b)
{
  using T = typename Vector::value_type;

  const std::int32_t local_size = a.bs() * a.index_map()->size_local();
  if (local_size != b.bs() * b.index_map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");

  T local = 0;
#if defined(USE_CUDA) || defined(USE_HIP)
  local = thrust::inner_product(thrust::device, a.array().begin(),
                                a.array().begin() + local_size,
                                b.array().begin(), T{0.0});
#else // CPU
  local = std::inner_product(a.array().begin(), a.array().begin() + local_size,
                             b.array().begin(), T{0.0});
#endif

  T result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM,
                a.index_map()->comm());
  return result;
}

} // namespace detail

/// Solve Ax = b using unpreconditioned CG
/// @param A Operator (with apply method) computes y=A.x
/// @param x Vector solution
/// @param b Vector RHS
/// @param max_iter Maximum number of iterations
/// @param rtol Tolerance (default 0.0 will enforce max_iter iterations)
template <typename Operator, typename Vector, typename S>
int cg_solve(Operator& A, Vector& x, const Vector& b, int max_iter,
             S rtol = 0.0)
{
  using T = typename Vector::value_type;
  static_assert(std::is_same<S, T>(), "Type mismatch");

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

    // MatVec
    // y = A.p;
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

// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "csr.hpp"
#include "forms.hpp"
#include "laplacian.hpp"
#include "mesh.hpp"
#include "poisson.h"
#include "util.hpp"
#include "vector.hpp"
#include <array>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <memory>
#include <mpi.h>
#include <random>

#if defined(USE_CUDA) || defined(USE_HIP)
#include <thrust/sequence.h>
#endif

using namespace dolfinx;
namespace po = boost::program_options;
using T = SCALAR_TYPE;

namespace
{
/// @brief TODO
/// @param ndofs
/// @param degree
/// @param mpi_size
/// @return
std::array<std::int64_t, 3> compute_num_cells(std::int64_t ndofs, int degree,
                                              int mpi_size)
{
  double nx_approx = (std::pow(ndofs * mpi_size, 1.0 / 3.0) - 1) / degree;
  std::int64_t n0 = static_cast<std::int64_t>(nx_approx);
  std::array<std::int64_t, 3> nx = {n0, n0, n0};

  // Try to improve fit to ndofs +/- 5 in each direction
  if (n0 > 5)
  {
    std::int64_t best_misfit
        = (n0 * degree + 1) * (n0 * degree + 1) * (n0 * degree + 1)
          - ndofs * mpi_size;
    best_misfit = std::abs(best_misfit);
    for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
    {
      for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
      {
        for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
        {
          std::int64_t misfit
              = (nx0 * degree + 1) * (ny0 * degree + 1) * (nz0 * degree + 1)
                - ndofs * mpi_size;
          if (std::abs(misfit) < best_misfit)
          {
            best_misfit = std::abs(misfit);
            nx = {nx0, ny0, nz0};
          }
        }
      }
    }
  }

  return nx;
}

/// @brief TODO
/// @tparam X
/// @param comm
/// @param n
/// @param geom_perturb_fact
/// @return
template <typename X>
mesh::Mesh<X> create_mesh(MPI_Comm comm, std::array<std::int64_t, 3> n,
                          X geom_perturb_fact)
{
  mesh::Mesh<T> mesh0
      = mesh::create_box<X>(comm, {{{0, 0, 0}, {1, 1, 1}}}, {n[0], n[1], n[2]},
                            mesh::CellType::hexahedron);

  if (geom_perturb_fact != 0.0)
  {
    double perturb_x = geom_perturb_fact * 1 / n[0];
    std::span geom_x = mesh0.geometry().x();
    std::mt19937 generator(42);
    std::uniform_real_distribution<T> distribution(-perturb_x, perturb_x);
    for (std::size_t i = 0; i < geom_x.size(); i += 3)
      geom_x[i] += distribution(generator);
  }

  // First order coordinate element
  auto element
      = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
          basix::element::family::P, basix::cell::type::hexahedron, 1,
          basix::element::lagrange_variant::gll_warped,
          basix::element::dpc_variant::unset, false));
  dolfinx::fem::CoordinateElement<T> celement(element);

  return benchdolfinx::ghost_layer_mesh(mesh0, celement);
}

} // namespace

int main(int argc, char* argv[])
{
  // Program options
  po::options_description desc("Options");
  desc.add_options()("help,h", "Print usage message")(
      "benchmark,b", po::value<std::size_t>()->default_value(0),
      "Test to run: 0=off, 1=correctness, 2=performance")(
      "ndofs", po::value<std::size_t>()->default_value(1000),
      "Number of degrees-of-freedom per MPI process")(
      "qmode", po::value<std::size_t>()->default_value(1),
      "Quadrature mode (0 or 1): qmode=0 has P+1 points in each direction,"
      "qmode=1 has P+2 points in each direction.")(
      "nreps", po::value<std::size_t>()->default_value(1000),
      "Number of repetitions")("order",
                               po::value<std::size_t>()->default_value(3),
                               "Polynomial degree \"P\" (2-7)")(
      "mat_comp", po::bool_switch()->default_value(false),
      "Compare result to matrix operator (slow with large ndofs)")(
      "geom_perturb_fact", po::value<T>()->default_value(0.125),
      "Randomly perturb the geometry (useful to check "
      "correctness)")("use_gauss", po::bool_switch()->default_value(false),
                      "Use Gauss quadrature rather than GLL quadrature");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << "DOLFINx benchmark\n-----------------\n";
    std::cout
        << "\n  Finite Element Operator Action Benchmark which computes\n";
    std::cout << "  the Laplacian operator on a cube mesh of hexahedral "
                 "elements.\n\n";
    std::cout << desc << std::endl;
    return 0;
  }

  std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  std::size_t nreps = vm["nreps"].as<std::size_t>();
  std::size_t order = vm["order"].as<std::size_t>();
  bool matrix_comparison = vm["mat_comp"].as<bool>();
  T geom_perturb_fact = vm["geom_perturb_fact"].as<T>();
  bool use_gauss = vm["use_gauss"].as<bool>();

  // Quadrature mode (qmode=0: nq = P + 1, qmode=1: nq = P + 2)
  std::size_t qmode = vm["qmode"].as<std::size_t>();
  if (qmode > 1)
    throw std::runtime_error("Invalid qmode.");

  const std::size_t benchmark = vm["benchmark"].as<std::size_t>();
  if (benchmark == 1)
  {
    ndofs = 15625;
    nreps = 1;
    order = 6;
    qmode = 1;
    matrix_comparison = true;
    geom_perturb_fact = 0.125;
    use_gauss = true;
  }

  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank(0), size(0);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Create mesh
    std::array<std::int64_t, 3> nx = compute_num_cells(ndofs, order, size);
    spdlog::info("Mesh shape: {}x{}x{}", nx[0], nx[1], nx[2]);
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        create_mesh<T>(comm, nx, geom_perturb_fact));

    // Finite element for higher-order discretisation
    auto element = basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
        mesh, std::make_shared<const fem::FiniteElement<T>>(element)));

    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    std::size_t ncells = mesh->topology()->index_map(tdim)->size_global();
    std::size_t ndofs_global = V->dofmap()->index_map->size_global();

    std::string fp_type = "float";
    if (std::is_same_v<T, float>)
      fp_type += "32";
    else if (std::is_same_v<T, double>)
      fp_type += "64";

    if (rank == 0)
    {
#if defined(USE_CUDA) || defined(USE_HIP)
      std::cout << device_information();
#endif
      std::cout << "-----------------------------------\n";
      std::cout << "Polynomial degree : " << order << "\n";
#ifndef USE_SLICED
      std::cout << "Sliced : no" << std::endl;
#else
      std::cout << "Sliced : yes" << std::endl;
      std::cout << "Slice size: " << SLICE_SIZE << std::endl;
#endif
      std::cout << "Number of ranks : " << size << "\n";
      std::cout << "Number of cells-global : " << ncells << "\n";
      std::cout << "Number of dofs-global : " << ndofs_global << "\n";
      std::cout << "Number of cells-rank : " << ncells / size << "\n";
      std::cout << "Number of dofs-rank : " << ndofs_global / size << "\n";
      std::cout << "Number of repetitions : " << nreps << "\n";
      std::cout << "Scalar Type: " << fp_type << "\n";
      std::cout << "XXXUse GLL: " << use_gauss << "\n";
      std::cout << "Foo: " << matrix_comparison << "\n";
      std::cout << "-----------------------------------\n";
      std::cout << std::flush;
    }

    Json::Value root;

    Json::Value& in_root = root["input"];
    in_root["p"] = (Json::UInt64)order;
    in_root["mpi_size"] = size;
    in_root["ncells"] = (Json::UInt64)ncells;
    in_root["ndofs"] = (Json::UInt64)ndofs_global;
    in_root["nreps"] = (Json::UInt64)nreps;
    in_root["scalar_type"] = fp_type;
    in_root["mat_comp"] = matrix_comparison;

    // Prepare and set Constants for the bilinear form
    spdlog::debug("Define forms");
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto a = std::make_shared<fem::Form<double>>(
        benchdolfinx::create_laplacian_form2(V, {{"c0", kappa}}, qmode,
                                             use_gauss, order));
    auto L = std::make_shared<fem::Form<double>>(
        benchdolfinx::create_laplacian_form1(V, {{"w0", f}}, qmode, use_gauss,
                                             order));

    spdlog::debug("Interpolate (rank {})", rank);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> out;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
          }
          return {out, {out.size()}};
        });

    int fdim = tdim - 1;
    spdlog::debug("Create f->c on {}", rank);
    topology->create_connectivity(fdim, tdim);
    spdlog::debug("Done f->c on {}", rank);

    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(1.3, bdofs, V);

#if defined(USE_CUDA) || defined(USE_HIP)

    // Copy data to GPU

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T>;

    // Input vector
    auto map = V->dofmap()->index_map;
    spdlog::info("Create device vector u");

    DeviceVector u(map, 1);
    u.set(T{0.0});

    // Output vector
    spdlog::info("Create device vector y");
    DeviceVector y(map, 1);
    y.set(T{0.0});

    // -----------------------------------------------------------------------------

    // Create matrix free operator
    spdlog::info("Create MatFreeLaplacian");
    dolfinx::common::Timer op_create_timer("% Create matfree operator");

    basix::quadrature::type quad_type
        = use_gauss ? basix::quadrature::type::gauss_jacobi
                    : basix::quadrature::type::gll;

    MatFreeLaplacian<T> op(*mesh, *V, *bc, order, qmode, T(kappa->value[0]),
                           quad_type, 0);

    op_create_timer.stop();

    la::Vector<T> b(map, 1);
    fem::assemble_vector(b.mutable_array(), *L);
    u.copy_from_host(b); // Copy data from host vector to device vector
    u.scatter_fwd_begin();

    // Matrix free
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nreps; ++i)
      op(u, y);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;

    T unorm = acc::norm(u);
    T ynorm = acc::norm(y);

    Json::Value& out_root = root["results"];
    if (rank == 0)
    {
      std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
      std::cout << "Mat-free action Gdofs/s: "
                << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;

      out_root["gdofs"] = ndofs_global * nreps / (1e9 * duration.count());

      std::cout << "Norm of u = " << unorm << std::endl;
      std::cout << "Norm of y = " << ynorm << std::endl;
    }

    if (matrix_comparison)
    {
      // Compare to assembling on CPU and copying matrix to GPU
      DeviceVector z(map, 1);
      z.set(T{0.0});

      acc::MatrixOperator<T> mat_op(a, {*bc});
      dolfinx::common::Timer mtimer("% CSR Matvec");
      for (int i = 0; i < nreps; ++i)
        mat_op(u, z);
      mtimer.stop();

      T unorm = acc::norm(u);
      T znorm = acc::norm(z);
      // Compute error
      DeviceVector e(map, 1);
      acc::axpy(e, T{-1.0}, y, z);
      T enorm = acc::norm(e);

      if (rank == 0)
      {
        std::cout << "Norm of u = " << unorm << "\n";
        std::cout << "Norm of z = " << znorm << "\n";
        std::cout << "Norm of error = " << enorm << "\n";
        std::cout << "Relative norm of error = " << enorm / znorm << "\n";
        out_root["error_norm"] = enorm;
      }

      // Compute error in diagonal computation
      DeviceVector mat_free_inv_diag(map, 1);
      op.get_diag_inverse(mat_free_inv_diag);
      DeviceVector mat_inv_diag(map, 1);
      mat_op.get_diag_inverse(mat_inv_diag);

      DeviceVector e_diag(map, 1);
      acc::axpy(e_diag, T{-1.0}, mat_inv_diag, mat_free_inv_diag);
      T dnorm = acc::norm(e_diag);

      if (rank == 0)
      {
        std::cout << "Norm of diagonal error = " << dnorm << "\n";
        out_root["diagonal_error_norm"] = dnorm;
      }
    }

#endif
    if (rank == 0)
    {
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "  ";
      const std::unique_ptr<Json::StreamWriter> writer(
          builder.newStreamWriter());
      std::ofstream strm("out.json", std::ofstream::out);
      writer->write(root, &strm);
    }

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}

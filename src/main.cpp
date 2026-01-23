// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "forms.hpp"
#include "laplacian_solver.hpp"
#include "mesh.hpp"
#include "util.hpp"
#include <array>
#include <basix/finite-element.h>
#include <boost/json.hpp>
#include <boost/program_options.hpp>
#include <dolfinx/fem/utils.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

#include <memory>
#include <mpi.h>

using namespace dolfinx;
namespace po = boost::program_options;
namespace json = boost::json;

namespace
{
/// @brief Run benchmark on a cube domain.
///
/// @param comm MPI Communicator.
/// @param nx Number of cells in direction (nx, ny, nz).
/// @param geom_perturb_fact Geometry perturbation factor
/// @param degree Polynomial degree of the elements.
/// @param qmode Quadrature mode (0 or 1).
/// @param nreps Number of repetitions of ....
/// @param use_gauss Use Gauss quadrature, rather than GLL
/// @param matrix_comparison Verify computation against the action of an
/// @param platform gpu or cpu
/// @param use_cg Use CG iterations
/// assembled CSR Matrix.
template <typename T>
json::value run_benchmark(MPI_Comm comm, std::array<std::int64_t, 3> nx,
                          double geom_perturb_fact, int degree, int qmode,
                          int nreps, bool use_gauss, bool matrix_comparison,
                          std::string platform, bool use_cg)
{
  int rank(0), size(0);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Create mesh
  spdlog::info("Mesh cells in each direction: {} x {} x {}", nx[0], nx[1],
               nx[2]);
  auto mesh = std::make_shared<mesh::Mesh<T>>(
      benchdolfinx::create_mesh<T>(comm, nx, geom_perturb_fact));

  // Finite element for higher-order discretisation
  auto element = basix::create_tp_element<T>(
      basix::element::family::P, basix::cell::type::hexahedron, degree,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(element)));

  auto topology = V->mesh()->topology_mutable();
  int tdim = topology->dim();

  // Prepare and set Constants for the bilinear form
  spdlog::debug("Define forms");
  auto kappa = std::make_shared<fem::Constant<T>>(2.0);
  auto f = std::make_shared<fem::Function<T>>(V);

  // TODO: Handle float type in generated code
  fem::Form<T> a = benchdolfinx::create_laplacian_form2<T>(
      V, {{"c0", kappa}}, qmode, use_gauss, degree);
  fem::Form<T> L = benchdolfinx::create_laplacian_form1<T>(
      V, {{"w0", f}}, qmode, use_gauss, degree);

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
  auto facets = mesh::exterior_facet_indices(*topology);
  auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
  fem::DirichletBC<T> bc(1.3, bdofs, V);

  benchdolfinx::BenchmarkResults results;
  if (platform == "cpu")
  {
    results = benchdolfinx::laplace_action_cpu<T>(
        a, L, bc, degree, qmode, kappa->value[0], nreps, use_gauss,
        matrix_comparison, use_cg);
  }
#if defined(USE_CUDA) || defined(USE_HIP)
  else if (platform == "gpu")
  {
    results = benchdolfinx::laplace_action_gpu<T>(
        a, L, bc, degree, qmode, kappa->value[0], nreps, use_gauss,
        matrix_comparison, use_cg);
  }
#endif
  else
    throw std::runtime_error("Invalid platform: " + platform);

  json::value output
      = {{"ncells_global", mesh->topology()->index_map(tdim)->size_global()},
         {"ndofs_global", dofmap->index_map->size_global()},
         {"mat_free_time", results.mat_free_time},
         {"u_norm", results.unorm},
         {"y_norm", results.ynorm},
         {"z_norm", results.znorm},
         {"gdof_per_second", dofmap->index_map->size_global() * nreps
                                 / (1e9 * results.mat_free_time)}};

  return output;
}
} // namespace

int main(int argc, char* argv[])
{
  std::string default_platform = "cpu";
#if defined(USE_CUDA) || defined(USE_HIP)
  default_platform = "gpu";
#endif

  // Define command line options
  po::options_description desc("Options");
  desc.add_options()("help,h", "Print usage message")
      //
      ("platform", po::value<std::string>()->default_value(default_platform),
       "Compute platform (cpu or gpu)")
      //
      ("float", po::value<std::size_t>()->default_value(64),
       "Float size (bits). 32 or 64.")
      //
      ("ndofs", po::value<std::size_t>()->default_value(1000),
       "Number of degrees-of-freedom per MPI process")
      //
      ("ndofs_global", po::value<std::size_t>()->default_value(0),
       "Number of global degrees-of-freedom")
      //
      ("qmode", po::value<std::size_t>()->default_value(1),
       "Quadrature mode (0 or 1): qmode=0 has P+1 points in each direction,"
       "qmode=1 has P+2 points in each direction.")
      //
      ("cg", po::bool_switch()->default_value(false),
       "Do CG iterations, rather than simple operator action")
      //
      ("nreps", po::value<std::size_t>()->default_value(1000),
       "Number of repetitions")
      //
      ("degree", po::value<std::size_t>()->default_value(3),
       "Polynomial degree \"P\" (2-7)")
      //
      ("mat_comp", po::bool_switch()->default_value(false),
       "Compare result to matrix operator (slow with large ndofs)")
      //
      ("geom_perturb_fact", po::value<double>()->default_value(0.0),
       "Randomly perturb the geometry (useful to check "
       "correctness)")
      //
      ("use_gauss", po::bool_switch()->default_value(false),
       "Use Gauss quadrature rather than GLL quadrature")
      //
      ("json", po::value<std::string>()->default_value(""),
       "Filename for JSON output");

  // Parse command line options
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  if (vm.count("ndofs") && !vm["ndofs"].defaulted() && vm.count("ndofs_global")
      && !vm["ndofs_global"].defaulted())
  {
    throw std::logic_error("Conflicting options 'ndofs' and 'ndofs_global'");
  }
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

  std::string platform = vm["platform"].as<std::string>();
  std::size_t float_size = vm["float"].as<std::size_t>();
  if (float_size != 32 and float_size != 64)
    throw std::runtime_error("Invalid float size. Must be 32 or 64.");
  std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  std::size_t ndofs_global = vm["ndofs_global"].as<std::size_t>();
  std::size_t nreps = vm["nreps"].as<std::size_t>();
  std::size_t degree = vm["degree"].as<std::size_t>();
  bool matrix_comparison = vm["mat_comp"].as<bool>();
  double geom_perturb_fact = vm["geom_perturb_fact"].as<double>();
  bool use_gauss = vm["use_gauss"].as<bool>();
  bool use_cg = vm["cg"].as<bool>();
  std::string json_filename = vm["json"].as<std::string>();

  if (use_cg and matrix_comparison)
    throw std::runtime_error("Cannot do matrix comparison with CG");

  // Quadrature mode (qmode=0: nq = P + 1, qmode=1: nq = P + 2)
  std::size_t qmode = vm["qmode"].as<std::size_t>();
  if (qmode > 1)
    throw std::runtime_error("Invalid qmode.");

  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank(0), size(0);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (vm["ndofs_global"].defaulted())
      ndofs_global = ndofs * size;
    else
      ndofs = ndofs_global / size;

    if (rank == 0)
    {
#if defined(USE_CUDA) || defined(USE_HIP)
      std::cout << benchdolfinx::get_device_information();
#endif
      std::cout << "-----------------------------------\n";
      std::cout << "Platform: " << platform << "\n";
      std::cout << "Polynomial degree : " << degree << "\n";
      std::cout << "Number of ranks : " << size << std::endl;
      std::cout << "Requested number of local DoFs : " << ndofs << std::endl;
      std::cout << "Number of repetitions : " << nreps << std::endl;
      std::cout << "Scalar Type: " << float_size << std::endl;
      std::cout << "Use Gauss-Jacobi: " << use_gauss << std::endl;
      std::cout << "Compare to matrix: " << matrix_comparison << std::endl;
      std::cout << "-----------------------------------" << std::endl;
      ;
      std::cout << std::flush;
    }

    json::value in_root = {{"p", degree},
                           {"mpi_size", size},
                           {"ndofs_local_requested", ndofs},
                           {"nreps", nreps},
                           {"scalar_size", float_size},
                           {"use_gauss", use_gauss},
                           {"mat_comp", matrix_comparison},
                           {"qmode", qmode},
                           {"cg", use_cg}};

    std::array<std::int64_t, 3> nx
        = benchdolfinx::compute_mesh_size(ndofs_global, degree);

    // Run benchmark
    json::value out_root;
    if (float_size == 32)
    {
      out_root = run_benchmark<float>(comm, nx, geom_perturb_fact, degree,
                                      qmode, nreps, use_gauss,
                                      matrix_comparison, platform, use_cg);
    }
    else if (float_size == 64)
    {
      out_root = run_benchmark<double>(comm, nx, geom_perturb_fact, degree,
                                       qmode, nreps, use_gauss,
                                       matrix_comparison, platform, use_cg);
    }
    else
    {
      throw std::runtime_error(
          std::format("Invalid float size {}. Must be 32 or 64.", float_size));
    }

    // Report performance data
    if (rank == 0 and !json_filename.empty())
    {
      json::value root = {{"input", in_root}, {"output", out_root}};
      std::string json_str = json::serialize(root);

      std::filesystem::path filename(json_filename);
      std::cout << "*** Writing output to:       " << filename << std::endl;
      std::cout << "*** Writing output to (abs): "
                << std::filesystem::absolute(filename) << std::endl;
      std::ofstream strm(filename, std::ofstream::out);
      strm << json_str << "\n";
    }
    else if (rank == 0)
    {
      std::cout << "*** Empty file: " << json_filename << std::endl;
    }

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}

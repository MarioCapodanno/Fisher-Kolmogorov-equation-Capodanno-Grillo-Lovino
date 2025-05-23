#include "FisherKolmogorov3D.hpp"
#include "ParameterReader.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Function to compute convergence order safely
double compute_convergence(const double err_current, const double err_prev,
                           const double h_current, const double h_prev) {
  if (err_prev == 0.0 || h_prev == 0.0 || err_current <= 0.0 ||
      h_current <= 0.0)
    return -1.0; // Indicates invalid convergence order
  return std::log(err_current / err_prev) / std::log(h_current / h_prev);
}

int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  std::string parameter_file = "../build/parameters.prm"; // Default value
  if (argc > 1) {
    parameter_file = argv[1]; // Use command line argument if provided
  } else {
    if (mpi_rank == 0) {
      std::cout << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
      std::cout << "No parameter file specified, using default: "
                << parameter_file << std::endl;
    }
  }

  // Read parameters
  ParameterHandler prm;
  SimulationParameters params;
  ParameterReader parameter_reader(prm);

  try {
    params = parameter_reader.read_parameters(parameter_file);
  } catch (const std::exception &e) {
    if (mpi_rank == 0)
      std::cerr << "Error reading parameter file: " << e.what() << std::endl;
    return 1;
  }

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  const std::vector<std::string> mesh_file_names = {
      "../mesh/mesh-cube-2.msh",  "../mesh/mesh-cube-4.msh",
      "../mesh/mesh-cube-8.msh",  "../mesh/mesh-cube-16.msh",
      "../mesh/mesh-cube-32.msh",
  };
  const std::vector<double> h_vector = {0.5, 0.25, 0.125, 0.0625, 0.03125};

  // Loop over mesh files and solve problems
  for (const auto &mesh : mesh_file_names) {
    FisherKolmogorov3D problem(mesh, params.dext, params.alpha, params.r,
                               params.T, params.deltat);

    problem.set_solver_parameters(
        params.max_newton_iterations, params.newton_tolerance,
        params.max_cg_iterations, params.cg_tolerance_factor);
    problem.setup();
    problem.solve();

    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }

  // Rank 0 handles output
  if (mpi_rank == 0) {
    std::cout << "===============================================" << std::endl;

    std::ofstream convergence_file("convergence.csv");
    if (!convergence_file) {
      std::cerr << "Error: could not open convergence.csv for writing."
                << std::endl;
      return 1;
    }
    convergence_file << "h,eL2,eH1" << std::endl;

    for (size_t i = 0; i < h_vector.size(); ++i) {
      convergence_file << h_vector[i] << "," << errors_L2[i] << ","
                       << errors_H1[i] << std::endl;

      std::cout << std::scientific << "h = " << std::setw(4)
                << std::setprecision(2) << h_vector[i]
                << " | eL2 = " << errors_L2[i];

      if (i > 0) {
        double p = compute_convergence(errors_L2[i], errors_L2[i - 1],
                                       h_vector[i], h_vector[i - 1]);
        if (p < 0)
          std::cout << " (n/a)";
        else
          std::cout << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
      } else {
        std::cout << " (  - )";
      }

      std::cout << " | eH1 = " << errors_H1[i];
      if (i > 0) {
        double p = compute_convergence(errors_H1[i], errors_H1[i - 1],
                                       h_vector[i], h_vector[i - 1]);
        if (p < 0)
          std::cout << " (n/a)";
        else
          std::cout << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
      } else {
        std::cout << " (  - )";
      }
      std::cout << "\n";
    }
  }
  return 0;
}
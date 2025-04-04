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

  // Ensure parameter file provided as command-line argument
  std::string parameter_file = "parameters.prm";

  // Read parameters
  ParameterHandler prm;
  ParameterReader parameter_reader(prm);
  try {
    parameter_reader.read_parameters(parameter_file);
  } catch (const std::exception &e) {
    if (mpi_rank == 0)
      std::cerr << "Error reading parameter file: " << e.what() << std::endl;
    return 1;
  }

  unsigned int degree;

  double T, deltat, alpha;
  unsigned int max_newton_iter, max_cg_iter;
  double newton_tolerance, cg_tolerance_factor;

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  const std::vector<std::string> mesh_file_names = {
      "../mesh/mesh-cube-2.msh",  "../mesh/mesh-cube-4.msh",
      "../mesh/mesh-cube-8.msh",  "../mesh/mesh-cube-16.msh",
      "../mesh/mesh-cube-32.msh",
  };
  const std::vector<double> h_vector = {0.5, 0.25, 0.125, 0.0625, 0.03125};

  // Read mesh and FE parameters
  prm.enter_subsection("Mesh & geometry parameters");
  degree = prm.get_integer("Degree");
  prm.leave_subsection();

  prm.enter_subsection("Physical constants");
  alpha = prm.get_double("Alpha coefficient");
  prm.leave_subsection();

  prm.enter_subsection("Time stepping parameters");
  T = prm.get_double("T");
  deltat = prm.get_double("deltat");
  prm.leave_subsection();

  prm.enter_subsection("Solver parameters");
  max_newton_iter = prm.get_integer("Max Newton iterations");
  newton_tolerance = prm.get_double("Newton tolerance");
  max_cg_iter = prm.get_integer("Max CG iterations");
  cg_tolerance_factor = prm.get_double("CG tolerance factor");
  prm.leave_subsection();

  // Loop over mesh files and solve problems
  for (const auto &mesh : mesh_file_names) {
    FisherKolmogorov3D problem(alpha, mesh, degree, T, deltat);

    problem.set_solver_parameters(max_newton_iter, newton_tolerance,
                                  max_cg_iter, cg_tolerance_factor);
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
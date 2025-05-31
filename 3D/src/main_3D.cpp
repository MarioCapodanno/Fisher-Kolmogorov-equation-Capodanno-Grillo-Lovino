#include "DiffusionTensor.hpp"
#include "FisherKolmogorov3D.hpp"
#include "ParameterReader.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Create a ConditionalOStream for printing only on rank 0
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // Create a TimerOutput object. We choose to print only at the end (summary),
  // and collect wall‐clock times.
  TimerOutput timer(MPI_COMM_WORLD,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times);

  // Read parameters
  ParameterHandler prm;
  SimulationParameters params;
  ParameterReader parameter_reader(prm);

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

  try {
    params = parameter_reader.read_parameters(parameter_file);
  } catch (const std::exception &e) {
    if (mpi_rank == 0)
      std::cerr << "Error reading parameter file: " << e.what() << std::endl;
    return 1;
  }

  const std::string mesh_file = "../mesh/brain-h3.0.msh";

  FisherKolmogorov3D problem(mesh_file, *params.diffusion_tensor, params.alpha,
                             params.r, params.T, params.deltat, pcout, timer);

  problem.set_solver_parameters(
      params.max_newton_iterations, params.newton_tolerance,
      params.max_cg_iterations, params.cg_tolerance_factor);
  problem.setup();
  problem.solve();

  return 0;
}
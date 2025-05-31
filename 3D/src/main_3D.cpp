#include "DiffusionTensor.hpp"
#include "FisherKolmogorov3D.hpp"
#include "ParameterReader.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Stream for std::cout output:
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  //Build an ofstream for writing the timer summary:
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  std::ostringstream fname;
  fname << "timer_summary_np" << n_procs << ".txt";
  std::ofstream timer_file(fname.str());
  if (mpi_rank == 0 && !timer_file.is_open())
  {
    std::cerr << "ERROR: could not open '" << fname.str() 
              << "' for writing\n";
    return 1;
  }

  // Build a ConditionalOStream for the timer output:
  ConditionalOStream timer_stream(timer_file, mpi_rank == 0);

  // Create a TimerOutput object to manage the timing of the program.
  TimerOutput timer(MPI_COMM_WORLD,
                    timer_stream,
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
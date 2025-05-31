#ifndef PARAMETER_READER_HPP
#define PARAMETER_READER_HPP

#include "DiffusionTensor.hpp"
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/numerics/data_out.h>
#include <string>

using namespace dealii;

struct SimulationParameters {
  double alpha; // Diffusion coefficient
  double dext;  // Diffusion coefficient for external term
  double daxn;  // Diffusion coefficient for axonal term

  double T;       // Final time
  double deltat;  // Time step size
  unsigned int r; // Polynomial degree

  unsigned int max_newton_iterations; // Max iterations for Newton's method
  double newton_tolerance;            // Tolerance for Newton's method
  unsigned int max_cg_iterations;     // Max iterations for CG solver
  double cg_tolerance_factor;         // Tolerance factor for CG solver

  std::string diffusion_tensor_type; // Type of diffusion tensor
  Point<3> tensor_center;            // Center point for directional tensors

  std::shared_ptr<DiffusionTensor<3>> diffusion_tensor;
};

/**
 * Class to read and manage program parameters from a parameter file.
 */
class ParameterReader : public Subscriptor {
public:
  /**
   * Constructor.
   * @param paramhandler Reference to a ParameterHandler object
   */
  ParameterReader(ParameterHandler &paramhandler);

  /**
   * Read parameters from a file.
   * @param parameter_file Path to the parameter file
   */
  SimulationParameters read_parameters(const std::string &parameter_file);

private:
  /**
   * Declare all parameters that can be specified in the parameter file.
   */
  void declare_parameters();

  /**
   * Create the appropriate diffusion tensor based on parameters.
   * @param params The simulation parameters
   */
  void create_diffusion_tensor(SimulationParameters &params);

  /**
   * Reference to the parameter handler that will manage the parameters.
   */
  ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler) {}

void ParameterReader::declare_parameters() {
  prm.enter_subsection("Mesh & geometry parameters");
  {
    prm.declare_entry("Mesh file", "../mesh/brain-h3.0.msh",
                      Patterns::Anything(), "Path to the mesh file");
    prm.declare_entry("Degree", "1", Patterns::Integer(1),
                      "Polynomial degree of finite element");
  }
  prm.leave_subsection();

  prm.enter_subsection("Physical constants");
  {
    prm.declare_entry("Dext", "4.0", Patterns::Double(0),
                      "Dext coefficient for diffusion term");
    prm.declare_entry("Daxn", "40.0", Patterns::Double(0),
                      "Daxn coefficient for diffusion term");
    prm.declare_entry("Alpha coefficient", "0.3", Patterns::Double(0),
                      "Alpha coefficient for reaction term");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time stepping parameters");
  {
    prm.declare_entry("T", "40.0", Patterns::Double(0),
                      "Final simulation time");

    prm.declare_entry("deltat", "2.0", Patterns::Double(0), "Time step size");

    prm.declare_entry(
        "Theta", "1.0", Patterns::Double(0.0, 1.0),
        "Theta value for the time-stepping method (0=explicit, 1=implicit)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solver parameters");
  {
    prm.declare_entry("Max Newton iterations", "1000", Patterns::Integer(1),
                      "Maximum number of Newton iterations");

    prm.declare_entry("Newton tolerance", "1e-6", Patterns::Double(0),
                      "Tolerance for Newton iterations");

    prm.declare_entry("Max CG iterations", "1000", Patterns::Integer(1),
                      "Maximum number of CG iterations");

    prm.declare_entry(
        "CG tolerance factor", "1e-6", Patterns::Double(0),
        "Tolerance factor for CG solver (multiplied by residual norm)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diffusion tensor parameters");
  {
    prm.declare_entry(
        "Diffusion tensor type", "Isotropic",
        Patterns::Selection("Isotropic|Radial|Circumferential"),
        "Type of diffusion tensor to use");

    prm.declare_entry("Center X", "55.0", Patterns::Double(),
                      "X coordinate of center point for directional tensors");

    prm.declare_entry("Center Y", "75.0", Patterns::Double(),
                      "Y coordinate of center point for directional tensors");

    prm.declare_entry("Center Z", "65.0", Patterns::Double(),
                      "Z coordinate of center point for directional tensors");
  }
  prm.leave_subsection();
}

void ParameterReader::create_diffusion_tensor(SimulationParameters &params) {
  if (params.diffusion_tensor_type == "Isotropic") {
    params.diffusion_tensor =
        std::make_shared<IsotropicDiffusionTensor<3>>(params.dext);
  } else if (params.diffusion_tensor_type == "Radial") {
    params.diffusion_tensor = std::make_shared<RadialDiffusionTensor<3>>(
        params.dext, params.daxn, params.tensor_center);
  } else if (params.diffusion_tensor_type == "Circumferential") {
    // For circumferential, create a 2D point with just Y and Z coordinates
    Point<2> cir_center = {params.tensor_center[1], params.tensor_center[2]};
    params.diffusion_tensor =
        std::make_shared<CircumferentialDiffusionTensor<3>>(
            params.dext, params.daxn, cir_center);
  } else {
    // Default to Radial if something goes wrong
    std::cerr << "Unknown diffusion tensor type: "
              << params.diffusion_tensor_type << ", defaulting to Radial"
              << std::endl;
    params.diffusion_tensor = std::make_shared<RadialDiffusionTensor<3>>(
        params.dext, params.daxn, params.tensor_center);
  }
}

SimulationParameters
ParameterReader::read_parameters(const std::string &parameter_file) {
  SimulationParameters params;

  try {
    // Check if file exists first
    std::ifstream file(parameter_file);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "Parameter file '" << parameter_file << "' not found.";
      throw std::runtime_error(oss.str());
    }
    file.close();

    declare_parameters();
    prm.parse_input(parameter_file);

    // Read parameters as before
    prm.enter_subsection("Mesh & geometry parameters");
    params.r = prm.get_integer("Degree");
    prm.leave_subsection();

    prm.enter_subsection("Physical constants");
    params.dext = prm.get_double("Dext");
    params.daxn = prm.get_double("Daxn");
    params.alpha = prm.get_double("Alpha coefficient");
    prm.leave_subsection();

    prm.enter_subsection("Time stepping parameters");
    params.T = prm.get_double("T");
    params.deltat = prm.get_double("deltat");
    prm.leave_subsection();

    prm.enter_subsection("Solver parameters");
    params.max_newton_iterations = prm.get_integer("Max Newton iterations");
    params.newton_tolerance = prm.get_double("Newton tolerance");
    params.max_cg_iterations = prm.get_integer("Max CG iterations");
    params.cg_tolerance_factor = prm.get_double("CG tolerance factor");
    prm.leave_subsection();

    prm.enter_subsection("Diffusion tensor parameters");
    params.diffusion_tensor_type = prm.get("Diffusion tensor type");
    params.tensor_center[0] = prm.get_double("Center X");
    params.tensor_center[1] = prm.get_double("Center Y");
    params.tensor_center[2] = prm.get_double("Center Z");
    prm.leave_subsection();

    create_diffusion_tensor(params);

  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << "Error with parameter file '" << parameter_file << "':\n"
        << "Please check your parameter file format.";
    throw std::runtime_error(oss.str());
  }

  return params;
}

#endif // PARAMETER_READER_HPP
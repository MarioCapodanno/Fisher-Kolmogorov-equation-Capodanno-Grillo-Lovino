#ifndef HEAT_NON_LINEAR_HPP
#define HEAT_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class FisherKolmogorov3D {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the d coefficient.
  class FunctionD : public TensorFunction<2, dim>
  {
  public:
    virtual Tensor<2, dim>
    value(const Point<dim> &/*p*/) const override
    {
      Tensor<2, dim> result;
      //const Tensor<1, dim> fiber_n = computeFiber(p);

      for (unsigned int i = 0; i < dim; ++i)
        {
          result[i][i] += 1.0; //dext;

          for (unsigned int j = 0; j < dim; ++j)
            result[i][j] += 0.0; //daxn * n * n;
        }

      return result;
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim> {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      double ex_s = std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                    std::cos(M_PI * p[2]) * std::exp(-get_time());
      return (3 * M_PI * M_PI - 1) * ex_s - 0.1 * ex_s * (1 - ex_s);
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim> {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      // cube mesh
      if (p[0] > 0.49 && p[0] < 0.51 && p[1] > 0.49 && p[1] < 0.51 &&
          p[2] > 0.49 && p[2] < 0.51) {
        return 1.0 * std::exp(-std::pow(300 * (p[0] - 0.5), 2) -
                              std::pow(300 * (p[1] - 0.5), 2) -
                              std::pow(300 * (p[2] - 0.5), 2));
      } else {
        return 0.0;
      }
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim> {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      return std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
             std::cos(M_PI * p[2]) * std::exp(-get_time());
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override {
      Tensor<1, dim> result;

      result[0] = -M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                  std::cos(M_PI * p[2]) * std::exp(-get_time());
      result[1] = -M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) *
                  std::cos(M_PI * p[2]) * std::exp(-get_time());
      result[2] = -M_PI * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                  std::sin(M_PI * p[2]) * std::exp(-get_time());

      return result;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  FisherKolmogorov3D(const double alpha_,
                     const std::string &mesh_file_name_,
                     const unsigned int &r_,
                     const double &T_, 
                     const double &deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , alpha(alpha_)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD) 
  {}

  // Set the parameters for the Newton method and CG solver.
  void set_solver_parameters(const unsigned int max_newton_iter,
                             const double newton_tol,
                             const unsigned int max_cg_iter,
                             const double cg_tol_factor);

  // Initialization.
  void setup();

  // Solve the problem.
  void solve();

  double compute_error(const VectorTools::NormType &norm_type);

protected:
  // Assemble the tangent problem.
  void assemble_system();

  // Solve the linear system associated to the tangent problem.
  void solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void solve_newton();

  // Output.
  void output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // d coefficient.
  FunctionD d;

  // alpha coefficient.
  const double alpha;

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial conditions.
  FunctionU0 u_0;

  // Exact solution.
  ExactSolution exact_solution;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Maximum number of Newton iterations.
  unsigned int max_newton_iterations;

  // Newton tolerance.
  double newton_tolerance;

  // Max CG iteration
  unsigned int max_cg_iterations;

  // CG tolerance factor.
  double cg_tolerance_factor;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;
};

#endif
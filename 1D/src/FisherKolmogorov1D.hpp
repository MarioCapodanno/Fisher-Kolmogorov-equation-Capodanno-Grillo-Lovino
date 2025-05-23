#ifndef NON_LINEAR_DIFFUSION_HPP
#define NON_LINEAR_DIFFUSION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class FisherKolmogorov1D {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Function for the initial condition.
  class FunctionU0 : public Function<dim> {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      if (p[0] == 0.0)
        return 0.1;
      return 0.0;
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim> {
  public:
    // Constructor.
    ExactSolution() {}

    // Evaluation.
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      return std::cos(M_PI * p[0]) * std::exp(-get_time());
    }

    // Gradient evaluation.
    // deal.II requires this method to return a Tensor (not a double), i.e. a
    // dim-dimensional vector. In our case, dim = 1, so that the Tensor will in
    // practice contain a single number. Nonetheless, we need to return an
    // object of type Tensor.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override {
      Tensor<1, dim> result;

      result[0] = -M_PI * std::sin(M_PI * p[0]) * std::exp(-get_time());

      return result;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  FisherKolmogorov1D(const unsigned int &N_, const unsigned int &r_,
                     const double &T_, const double &deltat_,
                     const double &theta_, const double &d_,
                     const double &alpha_)
      : T(T_), N(N_), r(r_), deltat(deltat_), theta(theta_), d(d_),
        alpha(alpha_) {}

  // Initialization.
  void setup();

  // Solve the problem.
  void solve();

protected:
  // Assemble the mass and stiffness matrices.
  void assemble_system();

  // Assemble the linear system.
  void solve_linear_system();

  // Solve the newton problem.
  void solve_newton();

  // Output.
  void output(const unsigned int &time_step) const;

  // Problem definition. ///////////////////////////////////////////////////////

  // Initial condition.
  FunctionU0 u_0;

  // Exact solution.
  ExactSolution exact_solution;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // N+1 is the number of elements.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Diffusion coefficient.
  const double d;

  // Reaction coefficient.
  const double alpha;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity;

  // Mass matrix M / deltat.
  SparseMatrix<double> mass_matrix;

  // Stiffness matrix A.
  SparseMatrix<double> stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  SparseMatrix<double> lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  SparseMatrix<double> rhs_matrix;

  // Right-hand side vector in the linear system.
  Vector<double> system_rhs;

  // System solution (including ghost elements).
  Vector<double> solution;

  // Newton system solution.
  Vector<double> delta;

  // Previous system solution.
  Vector<double> solution_old;

  // Residual vector.
  Vector<double> residual_vector;

  // Jacobian matrix.
  SparseMatrix<double> jacobian_matrix;
};

#endif
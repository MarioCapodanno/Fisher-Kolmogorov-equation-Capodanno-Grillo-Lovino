#include "FisherKolmogorov1D.hpp"

void FisherKolmogorov1D::setup() {
  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    GridGenerator::subdivided_hyper_cube(mesh, N + 1, -1.0, 1.0, true);

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
    GridOut grid_out;
    std::ofstream grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // "Distribute" the degrees of freedom. For a given finite element space,
    // initializes info on the control variables (how many they are, where
    // they are collocated, their "global indices", ...).
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity.copy_from(dsp);

    std::cout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);

    std::cout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution_old.reinit(dof_handler.n_dofs());
    delta.reinit(dof_handler.n_dofs());

    solution.reinit(dof_handler.n_dofs());
    solution_old = solution;
  }
}

void FisherKolmogorov1D::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  //
  std::vector<double> solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  //
  std::vector<double> solution_old_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_residual = 0.0;

    fe_values.get_function_values(solution, solution_loc);
    fe_values.get_function_gradients(solution, solution_gradient_loc);
    fe_values.get_function_values(solution_old, solution_old_loc);

    for (unsigned int q = 0; q < n_q; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Mass matrix.
          cell_matrix(i, j) += fe_values.shape_value(i, q) *
                               fe_values.shape_value(j, q) / deltat *
                               fe_values.JxW(q);

          // Non-linear stiffness matrix, first term.
          cell_matrix(i, j) += d * fe_values.shape_grad(i, q) *
                               fe_values.shape_grad(j, q) * fe_values.JxW(q);

          // Non-linear stiffness matrix, second term.
          cell_matrix(i, j) -= alpha * fe_values.shape_value(i, q) *
                               (1 - 2 * solution_loc[q]) *
                               fe_values.shape_value(j, q) * fe_values.JxW(q);
        }

        // Assemble the residual vector (with changed sign).

        // Time derivative term.
        cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) / deltat *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);

        // Diffusion term.
        cell_residual(i) -= d * fe_values.shape_grad(i, q) *
                            solution_gradient_loc[q] * fe_values.JxW(q);

        // Reaction term.
        cell_residual(i) += (alpha * solution_loc[q] * (1 - solution_loc[q])) *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);

    jacobian_matrix.add(dof_indices, cell_matrix);
    residual_vector.add(dof_indices, cell_residual);
  }
}

void FisherKolmogorov1D::solve_linear_system() {
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR preconditioner;
  preconditioner.initialize(
      jacobian_matrix,
      PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta, residual_vector, preconditioner);
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void FisherKolmogorov1D::solve_newton() {
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-6;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    std::cout << "  Newton iteration " << n_iter << "/" << n_max_iters
              << " - ||r|| = " << std::scientific << std::setprecision(6)
              << residual_norm << std::flush;

    if (residual_norm > residual_tolerance) {
      solve_linear_system();

      solution += delta;
    } else {
      std::cout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }
}

void FisherKolmogorov1D::output(const unsigned int &time_step) const {
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  data_out.build_patches();

  const std::string output_file_name = "output-" + std::to_string(deltat) +
                                       "_" + std::to_string(time_step) + ".vtu";
  std::ofstream output(output_file_name);
  data_out.write_vtu(output);
}

void FisherKolmogorov1D::solve() {
  std::cout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    std::cout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution);

    // Output the initial solution.
    output(0);
    std::cout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat) {
    time += deltat;
    ++time_step;

    // Store the old solution, so that it is available for assembly.
    solution_old = solution;

    std::cout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << std::fixed << time << std::endl;

    // At every time step, we invoke Newton's method to solve the non-linear
    // problem.
    solve_newton();

    output(time_step);

    std::cout << std::endl;
  }
}
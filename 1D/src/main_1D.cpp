#include <fstream>
#include <iostream>
#include <vector>

#include "FisherKolmogorov1D.hpp"

// Main function.
int main(int argc, char *argv[]) {
  // Default values:
  double d = 0.0001;
  double alpha = 1.0;

  // Check for command-line arguments.
  if (argc > 1)
    d = std::stod(argv[1]);
  if (argc > 2)
    alpha = std::stod(argv[2]);

  const unsigned int N = 199;
  const unsigned int degree = 1;
  const double T = 20.0;
  const double theta = 1;
  const double deltat = 0.1;

  FisherKolmogorov1D problem(N, degree, T, deltat, theta, d, alpha);

  problem.setup();
  problem.solve();

  return 0;
}
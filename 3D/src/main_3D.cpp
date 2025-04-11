#include "FisherKolmogorov3D.hpp"
#include "DiffusionTensor.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 1;

  const double T      = 40.0;
  const double deltat = 2.0;
  const double dext   = 4.0;
  const double daxn   = 40.0;
  const double alpha = 0.3;
  const Point<FisherKolmogorov3D::dim> rad_center = {50.0, 80.0, 40.0}; // approx Dorsal Motor Nucleus
  // const Point<FisherKolmogorov3D::dim - 1> cir_center = {80.0, 40.0}; // approx Dorsal Motor Nucleus (y,z)

  // IsotropicDiffusionTensor<FisherKolmogorov3D::dim> diffusionTensor(dext);
  RadialDiffusionTensor<FisherKolmogorov3D::dim> diffusionTensor(dext, daxn, rad_center);
  // CircumferentialDiffusionTensor<FisherKolmogorov3D::dim> diffusionTensor(dext, daxn, cir_center);
  // AxonBasedTensor<FisherKolmogorov3D::dim> diffusionTensor(dext, daxn, rad_center);
  FisherKolmogorov3D problem("../mesh/brain-h3.0.msh", diffusionTensor, alpha, degree, T, deltat);

  problem.setup();
  problem.solve();

  return 0;
}
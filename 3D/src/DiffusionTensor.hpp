#ifndef DIFFUSION_TENSOR_HPP
#define DIFFUSION_TENSOR_HPP

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

using namespace dealii;

template <unsigned int DIM>
Tensor<1, DIM> computeRad(const Point<DIM> &p, const Point<DIM> &rad_center);

template <unsigned int DIM>
Tensor<1, DIM> computeCirc(const Point<DIM> &p, const Point<2> &cir_center);

template <unsigned int DIM>
class DiffusionTensor : public TensorFunction<2, DIM> {
public:
  DiffusionTensor(double dext, double daxn) : dext(dext), daxn(daxn) {}

  Tensor<2, DIM> value(const Point<DIM> &p) const override {
    Tensor<2, DIM> result;
    const Tensor<1, DIM> fiber_dir = computeFiber(p);

    for (unsigned int i = 0; i < DIM; ++i) {
      result[i][i] += dext;

      for (unsigned int j = 0; j < DIM; ++j)
        result[i][j] += daxn * fiber_dir[i] * fiber_dir[j];
    }

    return result;
  }

protected:
  virtual Tensor<1, DIM> computeFiber(const Point<DIM> &p) const = 0;

private:
  const double dext, daxn;
};

template <unsigned int DIM>
class IsotropicDiffusionTensor : public DiffusionTensor<DIM> {
public:
  IsotropicDiffusionTensor(double dext) : DiffusionTensor<DIM>(dext, 0.0) {}

protected:
  Tensor<1, DIM> computeFiber(const Point<DIM> & /*p*/) const override {
    Tensor<1, DIM> fiber_dir;
    for (unsigned int i = 0; i < DIM; i++) {
      fiber_dir[i] = 0.0;
    }
    return fiber_dir;
  }
};

template <unsigned int DIM>
class RadialDiffusionTensor : public DiffusionTensor<DIM> {
public:
  RadialDiffusionTensor(double dext, double daxn,
                        const Point<DIM> &radial_center)
      : DiffusionTensor<DIM>(dext, daxn), rad_center(radial_center) {}

protected:
  Tensor<1, DIM> computeFiber(const Point<DIM> &p) const override {
    return computeRad<DIM>(p, rad_center);
  }

private:
  const Point<DIM> rad_center;
};

template <unsigned int DIM>
class CircumferentialDiffusionTensor : public DiffusionTensor<DIM> {
public:
  CircumferentialDiffusionTensor(double dext, double daxn,
                                 const Point<2> &cir_center)
      : DiffusionTensor<DIM>(dext, daxn), cir_center(cir_center) {}

protected:
  Tensor<1, DIM> computeFiber(const Point<DIM> &p) const override {
    return computeCirc<DIM>(p, cir_center);
  }

private:
  const Point<2> cir_center;
};

template <unsigned int DIM>
class AxonBasedTensor : public DiffusionTensor<DIM> {
public:
  AxonBasedTensor(double dext, double daxn, const Point<DIM> &axon_center)
      : DiffusionTensor<DIM>(dext, daxn), axon_center(axon_center) {}

protected:
  Tensor<1, DIM> computeFiber(const Point<DIM> &p) const override {
    auto distance_center = p.distance(axon_center);

    if constexpr (DIM != 3) {
      AssertThrow(false, ExcMessage("AxonBasedTensor::computeFiber() is only "
                                    "implemented for DIM==3."));
    } else {
      if (distance_center < axon_radius) {
        const Point<2> ax_center = {axon_center[0], axon_center[1]};
        return computeCirc<DIM>(p, ax_center);
      } else {
        return computeRad<DIM>(p, axon_center);
      }
    }
  }

private:
  const Point<DIM> axon_center;
  const double axon_radius = 10.0;
};

template <unsigned int DIM>
Tensor<1, DIM> computeRad(const Point<DIM> &p, const Point<DIM> &rad_center) {
  Tensor<1, DIM> fiber_dir;

  const double distance = p.distance(rad_center);
  for (unsigned int i = 0; i < DIM; i++) {
    fiber_dir[i] = ((p[i] - rad_center[i]) / distance + 1.e-6);
  }
  return fiber_dir;
};

template <unsigned int DIM>
Tensor<1, DIM> computeCirc(const Point<DIM> &p, const Point<2> &cir_center) {
  Tensor<1, DIM> fiber_dir = {};

  const double dy = p[1] - cir_center[0];
  const double dz = p[2] - cir_center[1];
  const double distance = std::sqrt(dy * dy + dz * dz);
  const double inv_factor = 1.0 / (distance + 1.e-6);

  // Set the circumferential direction components based on the axon center
  fiber_dir[0] = 0.0;
  fiber_dir[1] = -dz * inv_factor;
  fiber_dir[2] = dy * inv_factor;

  return fiber_dir;
};

#endif
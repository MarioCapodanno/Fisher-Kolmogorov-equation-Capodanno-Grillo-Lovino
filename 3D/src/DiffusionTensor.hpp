#ifndef DIFFUSION_TENSOR_HPP
#define DIFFUSION_TENSOR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <unsigned int DIM>
class DiffusionTensor : public TensorFunction<2, DIM>
{
public:
  DiffusionTensor(const double dext, const double daxn): dext(dext), daxn(daxn) {}

  virtual ~DiffusionTensor() = default;

  Tensor<2, DIM> value(const Point<DIM> &p) const override
  {
    Tensor<2, DIM> result;
    const Tensor<1, DIM> fiber_dir = compute_fiber(p);

    for (unsigned int i = 0; i < DIM; ++i)
    {
      result[i][i] += dext;
      for (unsigned int j = 0; j < DIM; ++j)
        result[i][j] += daxn * fiber_dir[i] * fiber_dir[j];
    }

    return result;
  }

protected:
  virtual Tensor<1, DIM> compute_fiber(const Point<DIM> &p) const = 0;

private:
  const double dext, daxn;
};


template <unsigned int DIM>
class IsotropicDiffusionTensor : public DiffusionTensor<DIM>
{
public:
  explicit IsotropicDiffusionTensor(const double dext)
    : DiffusionTensor<DIM>(dext, 0.0) {}

protected:
  Tensor<1, DIM> compute_fiber(const Point<DIM> & /*p*/) const override
  {
    return Tensor<1, DIM>();  // zero vector
  }
};


template <unsigned int DIM>
class RadialDiffusionTensor : public DiffusionTensor<DIM>
{
public:
  RadialDiffusionTensor(const double dext, const double daxn,
                        const Point<DIM> &radial_center)
    : DiffusionTensor<DIM>(dext, daxn), rad_center(radial_center) {}

protected:
  Tensor<1, DIM> compute_fiber(const Point<DIM> &p) const override
  {
    Tensor<1, DIM> fiber_dir;
    const double dist = p.distance(rad_center) + 1e-6;

    for (unsigned int i = 0; i < DIM; ++i)
      fiber_dir[i] = (p[i] - rad_center[i]) / dist;

    return fiber_dir;
  }

private:
  const Point<DIM> rad_center;
};


template <unsigned int DIM>
class CircumferentialDiffusionTensor : public DiffusionTensor<DIM>
{
public:
  CircumferentialDiffusionTensor(const double dext, const double daxn,
                                 const Point<2> &cir_center)
    : DiffusionTensor<DIM>(dext, daxn), cir_center(cir_center) {}

protected:
  Tensor<1, DIM> compute_fiber(const Point<DIM> &p) const override {
    static_assert(DIM >= 3, 
              "CircumferentialDiffusionTensor requires DIM >= 3");

    Tensor<1, DIM> fiber_dir = {};
    
    const double dy = p[1] - cir_center[0];
    const double dz = p[2] - cir_center[1];
    const double dist = std::sqrt(dy*dy + dz*dz) + 1e-6;

    fiber_dir[1] = -dz / dist;
    fiber_dir[2] =  dy / dist;

    return fiber_dir;
  }

private:
  const Point<2> cir_center;
};

#endif
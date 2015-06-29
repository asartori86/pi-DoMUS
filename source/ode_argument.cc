#include "../include/ode_argument.h"
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include "utilities.h"

template<typename VEC>
bool OdeArgument<VEC>::solver_should_restart(const double,
                                             const VEC &,
                                             const VEC &,
                                             const unsigned int,
                                             const double)
{
  return false;
}



template<typename VEC>
void OdeArgument<VEC>::jacobian(VEC &, const VEC &) const
{
  Assert(false, ExcPureFunctionCalled());
}



template<typename VEC>
void OdeArgument<VEC>::solve_jacobian_system(VEC &, const VEC &, const double) const
{
  Assert(false, ExcPureFunctionCalled());
}


template<typename VEC>
int OdeArgument<VEC>::setup_jacobian(const double,
                                     const VEC &,
                                     const VEC &,
                                     const double)
{
  Assert(false, ExcPureFunctionCalled());
  return 0;
}


template<typename VEC>
VEC &OdeArgument<VEC>::differential_components() const
{
  static shared_ptr<VEC> tmp = create_new_vector();
  static bool init = true;
  if (init == true)
    {
      tmp->add(1.0);
      init = false;
    }
  return (*tmp);
}

template<typename VEC>
VEC &OdeArgument<VEC>::get_local_tolerances() const
{
  static shared_ptr<VEC> tmp = create_new_vector();
  static bool init = true;
  if (init == true)
    {
      tmp->add(1.0);
      init = false;
    }
  return *tmp;
}

template class OdeArgument<Vector<double> >;
#ifdef DEAL_II_WITH_TRILINOS
template class OdeArgument<TrilinosWrappers::MPI::BlockVector>;
#endif
#include "pidomus.h"
#include "interfaces/non_consevative/navier_stokes.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  NavierStokes<dim> energy;
  piDoMUS<dim,spacedim,dim+1> navier_stokes_equation (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/navier_stokes_01.prm", "used_parameters.prm");

  navier_stokes_equation.run ();

  auto sol = navier_stokes_equation.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }
  return 0;
}

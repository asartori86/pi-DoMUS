#include "pidomus.h"
#include "interfaces/two_phase_flow.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;

  ThreePhaseFlow<dim,LADealII> f;
  piDoMUS<dim,dim,LADealII> solver ("pidomus",f);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/two_phase_flow_01.prm", "used_parameters.prm");


  solver.run ();

  auto sol = solver.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << sol[i] << std::endl;

  return 0;
}

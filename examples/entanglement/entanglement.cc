#include <pidomus.h>
#include "entanglement.h"

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  const int dim = 2;
  const int spacedim = 3;

  EntanglementInterface<dim,spacedim,LADealII> p;
  piDoMUS<dim,spacedim,LADealII> solver ("pidomus",p);
  ParameterAcceptor::initialize("entanglement.prm", "used_parameters.prm");

  solver.run ();

  return 0;
}

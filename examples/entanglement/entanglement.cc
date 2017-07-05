#include <pidomus.h>
#include "entanglement.h"

// typedef LADealII LAC;
typedef LATrilinos LAC;

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  const int dim = 2;
  const int spacedim = 3;

  EntanglementInterface<dim,spacedim,LAC> p;
  piDoMUS<dim,spacedim,LAC> solver ("pidomus",p);
  ParameterAcceptor::initialize("entanglement.prm", "used_parameters.prm");

  solver.run ();

  return 0;
}


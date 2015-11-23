#include "pidomus.h"
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
//
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/solution_transfer.h>
//#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal2lkit/utilities.h>

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <numeric>
#include <locale>
#include <string>
#include <math.h>

#include "lac/lac_initializer.h"

using namespace dealii;
using namespace deal2lkit;

/* ------------------------ PARAMETERS ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
void
piDoMUS<dim, spacedim, n_components, LAC>::
declare_parameters (ParameterHandler &prm)
{
  add_parameter(  prm,
                  &initial_global_refinement,
                  "Initial global refinement",
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &n_cycles,
                  "Number of cycles",
                  "3",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &max_time_iterations,
                  "Maximum number of time steps",
                  "10000",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &timer_file_name,
                  "Timer output file",
                  "timer.txt",
                  Patterns::FileName());


  add_parameter(  prm,
                  &adaptive_refinement,
                  "Adaptive refinement",
                  "true",
                  Patterns::Bool());


  add_parameter(  prm,
                  &use_direct_solver,
                  "Use direct solver if available",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &verbose,
                  "Print some useful informations about processes",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &overwrite_iter,
                  "Overwrite Newton's iterations",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &solver_tolerance,
                  "Linear solver tolerance",
                  "1e-8",
                  Patterns::Double(0.0));
}

template <int dim, int spacedim, int n_components, typename LAC>
void
piDoMUS<dim, spacedim, n_components, LAC>::parse_parameters_call_back()
{
  use_direct_solver &= (typeid(typename LAC::BlockMatrix) == typeid(dealii::BlockSparseMatrix<double>));
}


/* ------------------------ CONSTRUCTORS ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
piDoMUS<dim, spacedim, n_components, LAC>::piDoMUS (const Interface<dim, spacedim, n_components, LAC> &energy,
                                                    const MPI_Comm &communicator)
  :
  SundialsInterface<typename LAC::VectorType>(communicator),
  comm(communicator),
  energy(energy),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),
  timer_outfile("timer.txt"),
  tcout (timer_outfile,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),
  computing_timer (comm,
                   tcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),

  n_aux_matrices(energy.get_number_of_aux_matrices()),
  eh("Error Tables", energy.get_component_names(),
     print(std::vector<std::string>(n_components, "L2,H1"), ";")),

  pgg("Domain"),

  pgr("Refinement"),

  exact_solution("Exact solution"),
  initial_solution("Initial solution"),
  initial_solution_dot("Initial solution_dot"),

  data_out("Output Parameters", "vtu"),
  dae(*this),
  we_are_parallel(Utilities::MPI::n_mpi_processes(comm) > 1)
{
  for (unsigned int i=0; i<n_aux_matrices; ++i)
    {
      aux_matrix.push_back( SP( new typename LAC::BlockMatrix() ) );
      aux_matrix_sp.push_back( SP( new typename LAC::BlockSparsityPattern() ) );
    }
}


/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::setup_dofs (const bool &first_run)
{
  computing_timer.enter_section("Setup dof systems");
  std::vector<unsigned int> sub_blocks = energy.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

  mapping = energy.get_mapping(*dof_handler, solution);

  dofs_per_block.clear();
  dofs_per_block.resize(energy.n_blocks());

  DoFTools::count_dofs_per_block (*dof_handler, dofs_per_block,
                                  sub_blocks);

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler->n_dofs()
        << "(" << print(dofs_per_block, "+") << ")"
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  partitioning.resize(0);
  relevant_partitioning.resize(0);

  IndexSet relevant_set;
  {
    global_partitioning = dof_handler->locally_owned_dofs();
    for (unsigned int i = 0; i < energy.n_blocks(); ++i)
      partitioning.push_back(global_partitioning.get_view( std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                           std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));

    DoFTools::extract_locally_relevant_dofs (*dof_handler,
                                             relevant_set);

    for (unsigned int i = 0; i < energy.n_blocks(); ++i)
      relevant_partitioning.push_back(relevant_set.get_view(std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                            std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));
  }
  constraints.clear ();
  constraints.reinit (relevant_set);

  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);
  constraints.close ();

  ScopedLACInitializer initializer(dofs_per_block,
                                   partitioning,
                                   relevant_partitioning,
                                   comm);

  initializer(solution);
  initializer(solution_dot);
  if (we_are_parallel)
    {
      initializer.ghosted(distributed_solution);
      initializer.ghosted(distributed_solution_dot);
    }

  jacobian_matrix.clear();
  initializer(jacobian_matrix_sp,
              *dof_handler,
              constraints,
              energy.get_coupling());

  jacobian_matrix.reinit(jacobian_matrix_sp);

  if (energy.get_jacobian_preconditioner_flags() != update_default)
    {
      jacobian_preconditioner_matrix.clear();

      initializer(jacobian_preconditioner_matrix_sp,
                  *dof_handler,
                  constraints,
                  energy.get_preconditioner_coupling());

      jacobian_preconditioner_matrix.reinit(jacobian_preconditioner_matrix_sp);

      for (unsigned int i=0; i < n_aux_matrices; ++i)
        {
          aux_matrix[i]->clear();
          initializer(*aux_matrix_sp[i],
                      *dof_handler,
                      constraints,
                      energy.get_preconditioner_coupling());
          /*  energy.get_aux_matrix_coupling(i));*/
          aux_matrix[i]->reinit(*aux_matrix_sp[i]);
        }
    }

  if (first_run)
    {
      if (fe->has_support_points())
        {
          VectorTools::interpolate(*mapping, *dof_handler, initial_solution, solution);
          VectorTools::interpolate(*mapping, *dof_handler, initial_solution_dot, solution_dot);
        }
      else
        {
          const QGauss<dim> quadrature_formula(fe->degree + 1);
          //VectorTools::project(*mapping, *dof_handler, constraints, quadrature_formula, initial_solution, solution);
          //VectorTools::project(*mapping, *dof_handler, constraints, quadrature_formula, initial_solution_dot, solution_dot);
        }

    }
  computing_timer.exit_section();
}

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::assemble_jacobian_matrix (const double t,
    const typename LAC::VectorType &solution,
    const typename LAC::VectorType &solution_dot,
    const double alpha)
{
  computing_timer.enter_section ("   Assemble system jacobian");

  jacobian_matrix = 0;

  energy.set_time(t);
  constraints.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);

  constraints.close ();

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  typename LAC::VectorType tmp(solution);
  constraints.distribute(tmp);

  if (we_are_parallel)
    {
      distributed_solution = tmp;
      distributed_solution_dot = solution_dot;

      energy.initialize_data(distributed_solution,
                             distributed_solution_dot, t, alpha);
    }
  else
    {
      energy.initialize_data(tmp,
                             solution_dot, t, alpha);
    }


  auto local_copy = [ this ]
                    (const SystemCopyData & data)
  {
    this->constraints.distribute_local_to_global (data.local_matrix,
                                                  data.local_dof_indices,
                                                  this->jacobian_matrix);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         Scratch & scratch,
                         SystemCopyData & data)
  {
    this->energy.assemble_local_system(cell, scratch, data);
  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       Scratch(*mapping,
               *fe,
               quadrature_formula,
               energy.get_jacobian_flags(),
               face_quadrature_formula,
               energy.get_face_flags()),
       Assembly::CopyData::
       piDoMUSSystem<dim, spacedim> (*fe,n_aux_matrices));

  compress(jacobian_matrix, VectorOperation::add);

//  pcout << std::endl;

  // auto id = solution.locally_owned_elements();
  // for (unsigned int i = 0; i < id.n_elements(); ++i)
  //   {
  //     auto j = id.nth_index_in_set(i);
  //     if (constraints.is_constrained(j))
  //       jacobian_matrix.set(j, j, 1.0);
  //   }
  // compress(jacobian_matrix, VectorOperation::insert);

  computing_timer.exit_section();
}




template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::assemble_jacobian_preconditioner (const double t,
    const typename LAC::VectorType &solution,
    const typename LAC::VectorType &solution_dot,
    const double alpha)
{
  if (energy.get_jacobian_preconditioner_flags() != update_default)
    {
      computing_timer.enter_section ("   Build preconditioner");
      jacobian_preconditioner_matrix = 0;

      energy.set_time(t);
      constraints.clear();
      DoFTools::make_hanging_node_constraints (*dof_handler,
                                               constraints);

      energy.apply_dirichlet_bcs(*dof_handler, constraints);

      constraints.close ();


      const QGauss<dim> quadrature_formula(fe->degree + 1);
      const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

      typedef
      FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
      CellFilter;


      distributed_solution = solution;
      distributed_solution_dot = solution_dot;

      energy.initialize_data(distributed_solution,
                             distributed_solution_dot, t, alpha);


      auto local_copy = [this]
                        (const PreconditionerCopyData & data)
      {
        this->constraints.distribute_local_to_global (data.local_matrix,
                                                      data.local_dof_indices,
                                                      this->jacobian_preconditioner_matrix);
      };

      auto local_assemble = [ this ]
                            (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                             Scratch & scratch,
                             PreconditionerCopyData & data)
      {
        this->energy.assemble_local_preconditioner(cell, scratch, data);
      };



      WorkStream::
      run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                       dof_handler->begin_active()),
           CellFilter (IteratorFilters::LocallyOwnedCell(),
                       dof_handler->end()),
           local_assemble,
           local_copy,
           Scratch (*mapping,
                    *fe, quadrature_formula,
                    energy.get_jacobian_preconditioner_flags(),
                    face_quadrature_formula,
                    UpdateFlags(0)),
           Assembly::CopyData::
           piDoMUSPreconditioner<dim, spacedim> (*fe,n_aux_matrices));

      jacobian_preconditioner_matrix.compress(VectorOperation::add);
      computing_timer.exit_section();
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::update_all (const double t)
{
  energy.set_time(t);
  constraints.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);

  constraints.close ();
}


// aux matrices ////////////////////////////////////////////////////////////////

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::assemble_aux_matrices (const double t,
    const typename LAC::VectorType &solution,
    const typename LAC::VectorType &solution_dot,
    const double alpha)
{
  computing_timer.enter_section ("   Assemble aux matrices");
  update_all(t);
  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);
  distributed_solution = solution;
  distributed_solution_dot = solution_dot;

  energy.initialize_data(distributed_solution,
                         distributed_solution_dot, t, alpha);
  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;


  for (unsigned int i=0; i<n_aux_matrices; ++i)
    *(aux_matrix[i]) = 0;



  auto local_copy = [this]
                    (const PreconditionerCopyData & data)
  {

    for (unsigned int i=0; i<n_aux_matrices; ++i)
      this->constraints.distribute_local_to_global (data.local_matrices[i],
                                                    data.local_dof_indices,
                                                    *(this->aux_matrix[i]));
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         Scratch & scratch,
                         PreconditionerCopyData & data)
  {
    this->energy.assemble_local_aux_matrices(cell, scratch, data);
  };



  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       Scratch (*mapping,
                *fe, quadrature_formula,
                energy.get_jacobian_preconditioner_flags(),  /*energy.aux_matrix_flags(i),*/
                face_quadrature_formula,
                UpdateFlags(0)),
       Assembly::CopyData::
       piDoMUSPreconditioner<dim, spacedim> (*fe,n_aux_matrices));

  for (unsigned int i=0; i<n_aux_matrices; ++i)
    aux_matrix[i]->compress(VectorOperation::add);
  computing_timer.exit_section();
}


/* ------------------------ MESH AND GRID ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::refine_mesh ()
{
  computing_timer.enter_section ("   Mesh refinement");

  typedef TrilinosWrappers::MPI::BlockVector pVEC;
  typedef BlockVector<double> sVEC;

  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());
      KellyErrorEstimator<dim>::estimate (*dof_handler,
                                          QGauss < dim - 1 > (fe->degree + 1),
                                          typename FunctionMap<dim>::type(),
                                          distributed_solution,
                                          estimated_error_per_cell,
                                          ComponentMask(),
                                          0,
                                          0,
                                          triangulation->locally_owned_subdomain());

      pgr.mark_cells(estimated_error_per_cell, *triangulation);
    }


  if (typeid(typename LAC::VectorType) == typeid(pVEC))
    {
      parallel::distributed::SolutionTransfer<dim, pVEC> sol_tr(*dof_handler);
      parallel::distributed::SolutionTransfer<dim, pVEC> sol_dot_tr(*dof_handler);
      typename LAC::VectorType sol (distributed_solution);
      typename LAC::VectorType sol_dot (distributed_solution_dot);
      sol = solution;
      sol_dot = solution_dot;

      triangulation->prepare_coarsening_and_refinement();
      sol_tr.prepare_for_coarsening_and_refinement ((pVEC &)sol);
      sol_dot_tr.prepare_for_coarsening_and_refinement((pVEC &)sol_dot);
      if (adaptive_refinement)
        triangulation->execute_coarsening_and_refinement ();
      else
        triangulation->refine_global (1);

      setup_dofs(false);

      typename LAC::VectorType tmp (solution);
      typename LAC::VectorType tmp_dot (solution_dot);

      sol_tr.interpolate ((pVEC &)tmp);
      sol_dot_tr.interpolate ((pVEC &)tmp_dot);
      solution = tmp;
      solution_dot = tmp_dot;
    }
  else
    {
      SolutionTransfer<dim, sVEC> sol_tr(*dof_handler);
      SolutionTransfer<dim, sVEC> sol_dot_tr(*dof_handler);

      typename LAC::VectorType tmp (solution);
      typename LAC::VectorType tmp_dot (solution_dot);

      triangulation->prepare_coarsening_and_refinement();
      sol_tr.prepare_for_coarsening_and_refinement ((sVEC &)tmp);
      sol_dot_tr.prepare_for_coarsening_and_refinement((sVEC &)tmp_dot);
      if (adaptive_refinement)
        triangulation->execute_coarsening_and_refinement ();
      else
        triangulation->refine_global (1);

      setup_dofs(false);

      sol_tr.interpolate ((sVEC &)tmp, (sVEC &)solution);
      sol_dot_tr.interpolate ((sVEC &)tmp_dot, (sVEC &)solution_dot);
    }

  computing_timer.exit_section();
}

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::make_grid_fe()
{
  triangulation = SP(pgg.distributed(comm));
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  energy.postprocess_newly_created_triangulation(*triangulation);
  fe = SP(energy());
  triangulation->refine_global (initial_global_refinement);
}

/* ------------------------ OUTPUTS ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
typename LAC::VectorType
piDoMUS<dim, spacedim, n_components, LAC>::
get_solution()
{
  return solution;
};

/* ------------------------ RUN ------------------------ */

template <int dim, int spacedim, int n_components, typename LAC>
void piDoMUS<dim, spacedim, n_components, LAC>::run ()
{
//  if(timer_file_name != "")
//    timer_outfile.open(timer_file_name.c_str());
//  else
//    timer_outfile.open("/dev/null");
//

  for (current_cycle = 0; current_cycle < n_cycles; ++current_cycle)
    {
      if (current_cycle == 0)
        {
          make_grid_fe();
          setup_dofs(true);
        }
      else
        refine_mesh();

      constraints.distribute(solution);

      dae.start_ode(solution, solution_dot, max_time_iterations);
      eh.error_from_exact(*mapping, *dof_handler, distributed_solution, exact_solution);
    }

  eh.output_table(pcout);

  // std::ofstream f("errors.txt");
  computing_timer.print_summary();
  timer_outfile.close();
  // f.close();
}

/*** ODE Argument Interface ***/

template <int dim, int spacedim, int n_components, typename LAC>
shared_ptr<typename LAC::VectorType>
piDoMUS<dim, spacedim, n_components, LAC>::create_new_vector() const
{
  shared_ptr<typename LAC::VectorType> ret = SP(new typename LAC::VectorType(solution));
  return ret;
}


template <int dim, int spacedim, int n_components, typename LAC>
unsigned int
piDoMUS<dim, spacedim, n_components, LAC>::n_dofs() const
{
  return dof_handler->n_dofs();
}


template <int dim, int spacedim, int n_components, typename LAC>
void
piDoMUS<dim, spacedim, n_components, LAC>::output_step(const double  t,
                                                       const typename LAC::VectorType &solution,
                                                       const typename LAC::VectorType &solution_dot,
                                                       const unsigned int step_number,
                                                       const double /* h */ )
{
  computing_timer.enter_section ("Postprocessing");

  energy.set_time(t);
  constraints.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);

  constraints.close ();

  typename LAC::VectorType tmp(solution);
  constraints.distribute(tmp);
  distributed_solution = tmp;
  distributed_solution_dot = solution_dot;

  std::stringstream suffix;
  suffix << "." << current_cycle << "." << step_number;
  data_out.prepare_data_output( *dof_handler,
                                suffix.str());
  data_out.add_data_vector (distributed_solution, energy.get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list( energy.get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  data_out.add_data_vector (distributed_solution_dot, print(sol_dot_names, ","));

  data_out.write_data_and_clear(*mapping);

  computing_timer.exit_section ();
}



template <int dim, int spacedim, int n_components, typename LAC>
bool
piDoMUS<dim, spacedim, n_components, LAC>::solver_should_restart(const double t,
    const unsigned int step_number,
    const double h,
    typename LAC::VectorType &solution,
    typename LAC::VectorType &solution_dot)
{
  return false;
}


template <int dim, int spacedim, int n_components, typename LAC>
int
piDoMUS<dim, spacedim, n_components, LAC>::residual(const double t,
                                                    const typename LAC::VectorType &solution,
                                                    const typename LAC::VectorType &solution_dot,
                                                    typename LAC::VectorType &dst)
{
  computing_timer.enter_section ("Residual");
  energy.set_time(t);
  constraints.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);

  constraints.close ();

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  typename LAC::VectorType tmp(solution);
  constraints.distribute(tmp);

  distributed_solution = tmp;
  distributed_solution_dot = solution_dot;

  energy.initialize_data(distributed_solution,
                         distributed_solution_dot, t, 0.0);

  dst = 0;

  auto local_copy = [&dst, this] (const SystemCopyData & data)
  {
    this->constraints.distribute_local_to_global (data.double_residual,
                                                  data.local_dof_indices,
                                                  dst);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         Scratch & scratch,
                         SystemCopyData & data)
  {
    cell->get_dof_indices (data.local_dof_indices);
    this->energy.get_system_residual(cell, scratch, data, data.double_residual); /*data.double_residuals*/
    // apply conservative loads
    this->energy.apply_forcing_terms(cell, scratch, data, data.double_residual); /*data/double_residuals[0]*/

    if (cell->at_boundary())
      this->energy.apply_neumann_bcs(cell, scratch, data, data.double_residual); /*data/double_residuals[0]*/
  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       Scratch(*mapping,
               *fe,
               quadrature_formula,
               energy.get_jacobian_flags(),
               face_quadrature_formula,
               energy.get_face_flags()),
       SystemCopyData(*fe,n_aux_matrices));

//   constraints.distribute(dst);

  dst.compress(VectorOperation::add);

  auto id = solution.locally_owned_elements();
  for (unsigned int i = 0; i < id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        dst[j] = solution(j) - distributed_solution(j);
    }

  dst.compress(VectorOperation::insert);
  computing_timer.exit_section();
  return 0;
}


template <int dim, int spacedim, int n_components, typename LAC>
int
piDoMUS<dim, spacedim, n_components, LAC>::solve_jacobian_system(const double t,
    const typename LAC::VectorType &y,
    const typename LAC::VectorType &y_dot,
    const typename LAC::VectorType &,
    const double alpha,
    const typename LAC::VectorType &src,
    typename LAC::VectorType &dst) const
{
  computing_timer.enter_section ("   Solve system");
  set_constrained_dofs_to_zero(dst);

  typedef dealii::BlockSparseMatrix<double> sMAT;
  typedef dealii::BlockVector<double> sVEC;

  if (we_are_parallel == false &&
      use_direct_solver == true)
    {

      SparseDirectUMFPACK inverse;
      inverse.factorize((sMAT &) jacobian_matrix);
      inverse.vmult((sVEC &)dst, (sVEC &)src);

    }
  else
    {

      PrimitiveVectorMemory<typename LAC::VectorType> mem;
      SolverControl solver_control (jacobian_matrix.m(), solver_tolerance);

      SolverFGMRES<typename LAC::VectorType>
      solver(solver_control, mem,
             typename SolverFGMRES<typename LAC::VectorType>::AdditionalData(50, true));

      auto S_inv = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op);
      S_inv.vmult(dst, src);

      if (verbose)
        {
          if (!overwrite_iter)
            pcout << std::endl;
          pcout << " iterations:            "
                << solver_control.last_step();
          if (overwrite_iter)
            pcout << "               \r";
          else
            pcout << "               " << std::endl;
        }

    }

  set_constrained_dofs_to_zero(dst);

  computing_timer.exit_section();
  return 0;
}


template <int dim, int spacedim, int n_components, typename LAC>
int
piDoMUS<dim, spacedim, n_components, LAC>::setup_jacobian(const double t,
                                                          const typename LAC::VectorType &src_yy,
                                                          const typename LAC::VectorType &src_yp,
                                                          const typename LAC::VectorType &,
                                                          const double alpha)
{
  computing_timer.enter_section ("   Setup Jacobian");
  assemble_jacobian_matrix(t, src_yy, src_yp, alpha);
  if (use_direct_solver == false)
    {
      assemble_jacobian_preconditioner(t, src_yy, src_yp, alpha);
      assemble_aux_matrices(t, src_yy, src_yp, alpha);

      energy.compute_system_operators(*dof_handler,
                                      jacobian_matrix, jacobian_preconditioner_matrix,
                                      aux_matrix,
                                      jacobian_op, jacobian_preconditioner_op);
    }

  computing_timer.exit_section();

  return 0;
}



template <int dim, int spacedim, int n_components, typename LAC>
typename LAC::VectorType &
piDoMUS<dim, spacedim, n_components, LAC>::differential_components() const
{
  static typename LAC::VectorType diff_comps;
  diff_comps.reinit(solution);
  std::vector<unsigned int> block_diff = energy.get_differential_blocks();
  for (unsigned int i = 0; i < block_diff.size(); ++i)
    diff_comps.block(i) = block_diff[i];

  set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}



template <int dim, int spacedim, int n_components, typename LAC>
void
piDoMUS<dim, spacedim, n_components, LAC>::set_constrained_dofs_to_zero(typename LAC::VectorType &v) const
{
  for (unsigned int i = 0; i < global_partitioning.n_elements(); ++i)
    {
      auto j = global_partitioning.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        v[j] = 0;
    }
}

// template class piDoMUS<1>;

//template class piDoMUS<1,1,1>;
//template class piDoMUS<1,1,2>;
//template class piDoMUS<1,1,3>;
//template class piDoMUS<1,1,4>;

// template class piDoMUS<1,2,1>;
// template class piDoMUS<1,2,2>;
// template class piDoMUS<1,2,3>;
// template class piDoMUS<1,2,4>;

template class piDoMUS<2, 2, 1>;
template class piDoMUS<2, 2, 2>;
template class piDoMUS<2, 2, 3>;
template class piDoMUS<2, 2, 4>;
template class piDoMUS<2, 2, 5>;

template class piDoMUS<2, 2, 6>;
template class piDoMUS<2, 2, 7>;
template class piDoMUS<2, 2, 8>;

template class piDoMUS<2, 2, 1, LADealII>;
template class piDoMUS<2, 2, 2, LADealII>;
template class piDoMUS<2, 2, 3, LADealII>;
template class piDoMUS<2, 2, 4, LADealII>;
template class piDoMUS<2, 2, 5, LADealII>;
template class piDoMUS<2, 2, 6, LADealII>;
template class piDoMUS<2, 2, 7, LADealII>;
template class piDoMUS<2, 2, 8, LADealII>;


// template class piDoMUS<2,3,1>;
// template class piDoMUS<2,3,2>;
// template class piDoMUS<2,3,3>;
// template class piDoMUS<2,3,4>;



template class piDoMUS<3, 3, 1>;
template class piDoMUS<3, 3, 2>;
template class piDoMUS<3, 3, 3>;
template class piDoMUS<3, 3, 4>;
template class piDoMUS<3, 3, 5>;
template class piDoMUS<3, 3, 6>;
template class piDoMUS<3, 3, 7>;
template class piDoMUS<3, 3, 8>;


template class piDoMUS<3, 3, 1, LADealII>;
template class piDoMUS<3, 3, 2, LADealII>;
template class piDoMUS<3, 3, 3, LADealII>;
template class piDoMUS<3, 3, 4, LADealII>;
template class piDoMUS<3, 3, 5, LADealII>;
template class piDoMUS<3, 3, 6, LADealII>;
template class piDoMUS<3, 3, 7, LADealII>;
template class piDoMUS<3, 3, 8, LADealII>;

// template class piDoMUS<3>;;

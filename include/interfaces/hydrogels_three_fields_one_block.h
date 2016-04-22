/*! \addtogroup equations
 *  @{
 */

#ifndef _hydrogels_three_fields_one_block_h_
#define _hydrogels_three_fields_one_block_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_preconditioner_amg.h>
#include <deal2lkit/parsed_mapped_functions.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>

#include <deal.II/lac/trilinos_precondition.h>


#include<deal.II/lac/schur_complement.h>

#include "lac/lac_type.h"

#include <time.h>


template <int dim, int spacedim, typename LAC>
class HydroGelThreeFieldsOneBlock : public PDESystemInterface<dim,spacedim, HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>, LAC>
{
public:

  ~HydroGelThreeFieldsOneBlock() {}

  HydroGelThreeFieldsOneBlock();


  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values             |
            update_gradients          | /* this is the new entry */
            update_quadrature_points  |
            update_normal_vectors     |
            update_JxW_values);
  }


  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;

  virtual void connect_to_signals() const
  {
    auto &signals = this->get_signals();
    auto &pcout = this->get_pcout();
    if (this->wrinkling)
      {
        signals.postprocess_newly_created_triangulation.connect(
          [&](typename parallel::distributed::Triangulation<dim,spacedim> &tria)
        {
	  pcout << "applying random distortion to grid" <<std::endl;
          GridTools::distort_random(factor,tria,false);
        });
      }

    signals.fix_initial_conditions.connect(
	 [&](typename LAC::VectorType &sol,
	     typename LAC::VectorType &sol_dot)
	 {
        std::cout << "000000000000000000000000000000000000" <<std::endl;
        unsigned int dofs_per_cell = this->pfe()->dofs_per_cell;
        for (unsigned int i=0; i< dofs_per_cell; ++i)
        {
            pcout <<  this->pfe()->system_to_component_index(i).first;
        }


//        double cache = 0;
//	   double copy = 0;
	   

//	     typedef
//  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
//  CellFilter;

//  auto local_copy = [this]
//                    ()
//  {
//  };

//  auto local_assemble = [this]
//                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
//                         double &,
//                         double &)
//  {

//  };


//  WorkStream::
//  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
//                   this->get_dof_handler().begin_active()),
//       CellFilter (IteratorFilters::LocallyOwnedCell(),
//                   this->get_dof_handler().end()),
//       local_assemble,
//       local_copy,
//       cache,
//       copy);





	   
	 }
					   );
    
    signals.begin_make_grid_fe.connect(
      [&]()
    {
      pcout << "#########  make_grid_fe"<<std::endl;
    });
    signals.begin_setup_dofs.connect(
      [&]()
    {
      pcout << "#########  setup_dofs"<<std::endl;
    });
    signals.begin_refine_mesh.connect(
      [&]()
    {
      pcout << "#########  refine_mesh"<<std::endl;
    });
    signals.begin_setup_jacobian.connect(
      [&]()
    {
      pcout << "#########  setup_jacobian"<<std::endl;
    });
    signals.begin_residual.connect(
      [&]()
    {
      pcout << "#########  residual"<<std::endl;
    });
    signals.begin_solve_jacobian_system.connect(
      [&]()
    {
      pcout << "#########  solve_jacobian_system"<<std::endl;
    });
    signals.begin_refine_and_transfer_solutions.connect(
      [&]()
    {
      pcout << "#########  refine_and_transfer_solutions"<<std::endl;
    });
    signals.begin_assemble_matrices.connect(
      [&]()
    {
      pcout << "#########  assemble_matrices"<<std::endl;
    });
    signals.begin_solver_should_restart.connect(
      [&]()
    {
      pcout << "#########  solver_should_restart"<<std::endl;
    });

  }

private:
  double T;
  double Omega;
  double G;
  double chi;
  double l0;

  double mu0;
  double l02;
  double l03;
  double l0_3;
  double l0_6;
  const double R=8.314;


  double factor;
  bool wrinkling;

  mutable ParsedAMGPreconditioner U_prec;
  mutable ParsedAMGPreconditioner c_prec_amg;

  mutable shared_ptr<TrilinosWrappers::PreconditionSSOR> p_prec_ssor;

  mutable shared_ptr<TrilinosWrappers::PreconditionBlockJacobi> preconditionblockjacobi;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionBlockSOR> preconditionblocksor;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionBlockSSOR> preconditionblockssor;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionBlockwiseDirect> preconditionblockwisedirect;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionChebyshev> preconditionchebyshev;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionIC> preconditionic;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionIdentity> preconditionidentity;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionILU> preconditionilu;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionILUT> preconditionilut;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditionjacobi;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionSOR> preconditionsor;
 
  mutable shared_ptr<TrilinosWrappers::PreconditionSSOR> preconditionssor;
  
  unsigned int it_c_lumped;
  unsigned int it_s_approx;
  unsigned int it_s;

  double gamma;
  mutable  ParsedMappedFunctions<spacedim> nitsche;


};

template <int dim, int spacedim, typename LAC>
HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>::HydroGelThreeFieldsOneBlock() :
  PDESystemInterface<dim,spacedim,HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>, LAC>("Free Swelling Three Fields",
      dim+2,2,
      "FESystem[FE_Q(1)^d-FE_DGPMonomial(0)-FE_DGPMonomial(0)]",
      "u,u,u,u,u","0"),
  U_prec("AMG for U"),
  nitsche("Nitsche boundary conditions",
          this->n_components,
          this->get_component_names(),
          "" /* do nothing by default */
         )

{}


template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &,
                       bool compute_only_system_terms) const
{
  EnergyType alpha = 0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar concentration(dim);
  const FEValuesExtractors::Scalar pressure(dim+1);
  {
    auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

    auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);

    auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

    const unsigned int n_q_points = ps.size();

    auto &JxW = fe_cache.get_JxW_values();

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const Tensor<2, dim, EnergyType>  &F = Fs[q];
        const Tensor<2, dim, EnergyType>   C = transpose(F)*F;
        const EnergyType &c = cs[q];
        const EnergyType &p = ps[q];


        const EnergyType I = trace(C);
        const EnergyType J = determinant(F);


        EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                           + (l0_3*R*T/Omega)*(
                             (Omega*l03*c)*std::log(
                               (Omega*l03*c)/(1.+Omega*l03*c)
                             )
                             + chi*(
                               (Omega*l03*c)/(1.+Omega*l03*c)
                             )
                           )

                           - (mu0)*c - p*(J-l0_3-Omega*c)
                         ) ;

        energies[0] += psi*JxW[q];

        if (!compute_only_system_terms)
          {
            EnergyType pp = 0.5*p*p ;
            energies[1] += pp*JxW[q];
          }
      }
  }

  /// (quasi) automatic nitsche bcs
  {
    EnergyType dummy;
    const double h = cell->diameter();

    for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        unsigned int face_id = cell->face(face)->boundary_id();
        if (cell->face(face)->at_boundary() && nitsche.acts_on_id(face_id))
          {
            this->reinit(dummy, cell, face, fe_cache);
            auto &ps = fe_cache.get_values("solution", "p", pressure, dummy);
            auto &us = fe_cache.get_values("solution", "u", displacement, dummy);
            auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, dummy);


            auto &fev = fe_cache.get_current_fe_values();
            auto &q_points = fe_cache.get_quadrature_points();
            auto &JxW = fe_cache.get_JxW_values();

            for (unsigned int q=0; q<q_points.size(); ++q)
              {
                auto &u = us[q];
                auto &p = ps[q];
                const Tensor<1,spacedim> n = fev.normal_vector(q);

                auto &F = Fs[q];
                Tensor<2,dim,EnergyType> C = transpose(F)*F;
                Tensor<2,dim,EnergyType> F_inv=invert(F);

                EnergyType Ic = trace(C);
                EnergyType J = determinant(F);
                EnergyType lnJ = std::log (J);


                Tensor<2,dim,EnergyType> S = invert(transpose(F));
                S *= -p*J;
                S += G/l0*F;

                // update time for nitsche_bcs
                nitsche.set_time(this->get_current_time());

                // get mapped function acting on this face_id
                Vector<double> func(this->n_components);
                nitsche.get_mapped_function(face_id)->vector_value(q_points[q], func);

                Tensor<1,spacedim> u0;

                for (unsigned int c=0; c<spacedim; ++c)
                  u0[c] = func[c];

                energies[0] +=(
                                (S*n)*(u-u0)

                                + (1.0/(2.0*gamma*h))*(u-u0)*(u-u0)
                              )*JxW[q];


              }// end loop over quadrature points

            break;

          } // endif face->at_boundary

      }// end loop over faces
  }


}

template <int dim, int spacedim, typename LAC>
void HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>, LAC>::declare_parameters(prm);
  this->add_parameter(prm, &T, "T", "298.0", Patterns::Double(0.0));
  this->add_parameter(prm, &Omega, "Omega", "1e-5", Patterns::Double(0.0));
  this->add_parameter(prm, &chi, "chi", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &l0, "l0", "1.5", Patterns::Double(1.0));
  this->add_parameter(prm, &G, "G", "10e3", Patterns::Double(0.0));
  this->add_parameter(prm, &it_c_lumped, "iteration c lumped", "10", Patterns::Integer(1));
  this->add_parameter(prm, &it_s_approx, "iteration s approx", "10", Patterns::Integer(1));
  this->add_parameter(prm, &it_s, "iteration s", "10", Patterns::Integer(1));

  this->add_parameter(prm, &gamma, "Gamma", "0.001", Patterns::Double(0));
  this->add_parameter(prm, &factor, "distortion factor", "1e-4", Patterns::Double(0.0));
  this->add_parameter(prm, &wrinkling, "distort triangulation", "false", Patterns::Bool());

}

template <int dim, int spacedim, typename LAC>
void HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>::parse_parameters_call_back ()
{
  l02 = l0*l0;
  l03 = l02*l0;
  l0_3 = 1./l03;
  l0_6 = 1./(l03*l03);

  mu0 = R*T*(std::log((l03-1.)/l03) + l0_3 + chi*l0_6) + G*Omega/l0;
}


template <int dim, int spacedim, typename LAC>
void
HydroGelThreeFieldsOneBlock<dim,spacedim,LAC>::
compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
                         LinearOperator<LATrilinos::VectorType> &system_op,
                         LinearOperator<LATrilinos::VectorType> &prec_op,
                         LinearOperator<LATrilinos::VectorType> &) const
{


  
  auto &pcout = this->get_pcout();
  clock_t inizio = clock();


  double tempo;

  //  auto &fe = this->pfe;


  U_prec.initialize_preconditioner(matrices[0]->block(0,0));

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();
  pcout << "u amg " << tempo << " seconds" << std::endl;
  //      }



  // SYSTEM MATRIX:
  auto A   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0) );


  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "system " << tempo << " seconds" << std::endl;

  auto A_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0), U_prec);

  auto P_i = A_inv;

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P_i}} ,
    }
  });

    tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();
  pcout << "prec_op " << tempo << " seconds" << std::endl;

}



#endif
/*! @} */

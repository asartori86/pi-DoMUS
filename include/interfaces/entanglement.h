#ifndef _pidoums_entanglement_h_
#define _pidoums_entanglement_h_

#include "pde_system_interface.h"
#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/parsed_preconditioner/amg.h>


template <int dim, int spacedim, typename LAC=LADealII>
class EntanglementInterface : public PDESystemInterface<dim,spacedim, EntanglementInterface<dim,spacedim,LAC>, LAC>
{

public:
  ~EntanglementInterface () {};
  EntanglementInterface ();

  /* void declare_parameters (ParameterHandler &prm); */
  /* void parse_parameters_call_back (); */


  virtual UpdateFlags get_cell_update_flags() const
  {
    return (update_values             |
            update_gradients          |
            update_quadrature_points  |
            // update_normal_vectors     |
            update_jacobians          |
            update_JxW_values);
  }



  // interface with the PDESystemInterface :)


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


//  template<int dim, int spacedim, typename LAC>
  virtual void
  estimate_error_per_cell(Vector<float> &estimated_error) const
  {
    //  std::vector<bool> v= {false,false,true};
    const DoFHandler<dim,spacedim> &dof = this->get_dof_handler();
    KellyErrorEstimator<dim,spacedim>::estimate (this->get_kelly_mapping(),
                                                 dof,
                                                 QGauss <dim-1> (dof.get_fe().degree + 1),
                                                 typename FunctionMap<spacedim>::type(),
                                                 this->get_locally_relevant_solution(),
                                                 estimated_error,
                                                 ComponentMask(),
                                                 0,
                                                 0,
                                                 dof.get_triangulation().locally_owned_subdomain());
  }

  virtual void connect_to_signals() const
  {
//    // first of all we get the struct Signals from pidomus
//    auto &signals = this->get_signals();
//    auto &pcout = this->get_pcout();
//signals.solution_preprocessing.connect(
//          [&](FEValuesCache<dim,spacedim> &cache)
//    {

//      cac

//    }


//        );

//    // we can connect calling .connect( and defining a lambda
//    signals.fix_initial_conditions.connect(
//      [this](typename LAC::VectorType &, typename LAC::VectorType &)
//    {
//      std::cout << "ciao mondo" << std::endl;
//    }
//    );

    // or we can define a lambda first
//    auto l =  [this](typename LAC::VectorType &, typename LAC::VectorType &)
//    {
//        auto tria=this->get_triangulation();
//        GridTools::remove_anisotropy(const_cast<Triangulation<dim,spacedim>(*tria));
//      std::cout << "ho raffinato" << std::endl;
//    };

//    // and then attach the just defined lambda
//    signals.fix_solutions_after_refinement.connect(l);


//    // herebelow, we connect to all the begin_* signals available in piDoMUS
//    auto &pcout = this->get_pcout();
//    signals.begin_make_grid_fe.connect(
//      [&]()
//    {
//      pcout << "#########  make_grid_fe"<<std::endl;
//    });
//    signals.begin_setup_dofs.connect(
//      [&]()
//    {
//      pcout << "#########  setup_dofs"<<std::endl;
//    });
//    signals.begin_refine_mesh.connect(
//      [&]()
//    {
//      pcout << "#########  refine_mesh"<<std::endl;
//    });
//    signals.begin_setup_jacobian.connect(
//      [&]()
//    {
//      pcout << "#########  setup_jacobian"<<std::endl;
//    });
//    signals.begin_residual.connect(
//      [&]()
//    {
//      pcout << "#########  residual"<<std::endl;
//    });
//    signals.begin_solve_jacobian_system.connect(
//      [&]()
//    {
//      pcout << "#########  solve_jacobian_system"<<std::endl;
//    });
//    signals.begin_refine_and_transfer_solutions.connect(
//      [&]()
//    {
//      pcout << "#########  refine_and_transfer_solutions"<<std::endl;
//    });
//    signals.begin_assemble_matrices.connect(
//      [&]()
//    {
//      pcout << "#########  assemble_matrices"<<std::endl;
//    });
//    signals.begin_solver_should_restart.connect(
//      [&]()
//    {
//      pcout << "#########  solver_should_restart"<<std::endl;
//    });

  }


  mutable ParsedAMGPreconditioner U_prec;

};

template <int dim, int spacedim, typename LAC>
EntanglementInterface<dim,spacedim, LAC>::
EntanglementInterface():
  PDESystemInterface<dim,spacedim,EntanglementInterface<dim,spacedim,LAC>, LAC >("Entanglement",
      3,1,
      "FESystem[FE_Q<2,3>(1)^3]",
      "u,u,u","1")
{}

namespace d2kinternal
{
  template <int dim, int spacedim, typename Number>
  inline
  Number determinant (const DerivativeForm<1,dim,spacedim,Number> &DF)
  {
    const DerivativeForm<1,spacedim,dim,Number> DF_t = DF.transpose();
    Tensor<2,dim,Number> G; //First fundamental form
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        G[i][j] = DF_t[i] * DF_t[j];

    return ( sqrt(determinant(G)) );
  }
}


template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
EntanglementInterface<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool ) const
{
  const FEValuesExtractors::Vector u(0);

  EnergyType et = 0; // dummy number to define the type of variables
  ResidualType rt=0;
  double dut=0;
  this->reinit (et, cell, fe_cache);
//  fe_cache.cache_local_solution_vector("solution_dot",this->get_locally_relevant_solution_dot(),rt);
//  fe_cache.cache_local_solution_vector("explicit_solution",this->get_locally_relevant_explicit_solution(),dut);
  auto &grad_eus = fe_cache.get_gradients("explicit_solution", "grad_u", u, dut);
  auto &us = fe_cache.get_values("solution", "u", u, et);
  auto &ues = fe_cache.get_values("explicit_solution", "uex", u , dut);
//  auto &uts = fe_cache.get_values("solution_dot", "ut", u, rt);
  auto &grad_us = fe_cache.get_gradients("solution", "grad_u",u, et);

  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();
  auto &jacs = fe_cache.get_current_fe_values().get_jacobians();
  auto fev = dynamic_cast<const FEValues<dim,spacedim> *>(&(fe_cache.get_current_fe_values()));
  Assert(fev != nullptr, ExcInternalError());
  auto &mfev = fe_cache.get_current_fe_values();
  auto &points = fev->get_quadrature_points();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &uz = us[q][spacedim-1];
      auto &grad_u = grad_us[q];
      auto &jac   = jacs[q];
      auto &p     = points[q];

      DerivativeForm<1,dim,spacedim,EnergyType> X;
      for (unsigned int a=0; a<dim; ++a)
        for (unsigned int i=0; i<spacedim; ++i)
          {
            X[i][a] = jac[i][a];
            for (unsigned int j=0; j<spacedim; ++j)
              X[i][a] += jac[j][a]*grad_u[i][j];
          }

      EnergyType Y=0,P=0;
      for (unsigned int a=0; a<dim; ++a)
        for (unsigned int i=0; i<spacedim; ++i)
          {
            Y=0;
            for (unsigned int j=0; j<spacedim; ++j)
              Y += jac[j][a]*grad_u[i][j];
            P+=Y*Y;
          }
//        static const double eps = 1e-1;
      EnergyType z = p[spacedim-1]+uz;
      EnergyType psi = d2kinternal::determinant(X)/(z*z);
      EnergyType psi_ = d2kinternal::determinant(X);

      auto grad_u_ = grad_u;
      const double ee = 1e-1;
      grad_u_[2][0]*=ee;
      grad_u_[2][1]*=ee;
      grad_u_[2][2]*=ee;
      grad_u_[0][2]*=ee;
      grad_u_[1][2]*=ee;

      auto grad_eu_ = grad_eus[q];

      grad_eu_[2]=0;
      grad_eu_[0][2]=0;
      grad_eu_[1][2]=0;
      EnergyType stab = scalar_product(grad_u,grad_u);
      double W = fev->get_quadrature().weight(q);
      EnergyType penalty = ues[q][2]*uz;

      energies[0] += (psi*W  /*+0.001*P*W*/ /* +0.5*stab*JxW[q]*/ +scalar_product(grad_u_,grad_u_)*JxW[q] /*+penalty*JxW[q]*/);

      et += psi*W;
      dut += SacadoTools::to_double(psi_)*W;
//        static const unsigned int size =local_residuals[0].size();
//        ResidualType pval = P.val();
//  for (unsigned int i=0; i<size; ++i)
//    {
//      auto v = mfev[u].value(i,q);
//      local_residuals[0][i] += (1000.*uts[q]*v);
//    }


    }


  AnyData &cache = fe_cache.get_cache();

  if (cache.have("area"))
    {
      cache.get<double>("area") = SacadoTools::to_double(et);
      cache.get<double>("area_eucl") = dut;
//        std::cout << "interface " << SacadoTools::to_double(et) << std::endl;
    }




}

template <int dim, int spacedim, typename LAC>
void
  EntanglementInterface<dim,spacedim,LAC>::compute_system_operators(
  const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
  LinearOperator<LATrilinos::VectorType> &system_op,
  LinearOperator<LATrilinos::VectorType> &prec_op,
  LinearOperator<LATrilinos::VectorType> &) const
{

  U_prec.initialize_preconditioner(matrices[0]->block(0,0));

  auto A_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0), U_prec);

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );



  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A_inv}} ,
    }
  });
}



#endif

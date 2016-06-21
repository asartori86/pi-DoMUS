/*! \addtogroup equations
 *  @{
 */

#ifndef _two_phase_flow_h
#define _two_phase_flow_h

#include "pde_system_interface.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>



#include<deal.II/lac/schur_complement.h>

#include<deal2lkit/sacado_tools.h>

#include "lac/lac_type.h"

#include <time.h>



template <int dim, typename LAC>
class TwoPhaseFlow : public PDESystemInterface<dim, dim, TwoPhaseFlow<dim,LAC>, LAC>
{
public:

  ~TwoPhaseFlow() {}

  TwoPhaseFlow();

  void declare_parameters (ParameterHandler &prm);

//  void set_matrix_couplings(std::vector<std::string> &couplings) const;

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim>::active_cell_iterator &cell,
                              FEValuesCache<dim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


//  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
//                                LinearOperator<LATrilinos::VectorType> &,
//                                LinearOperator<LATrilinos::VectorType> &) const;



private:
    const double lambda = 1.0e-2;
  double rho1;
  double rho2;
  double eta1;
  double eta2;
  double eps;
  double M0;

  template <typename Number>
  Number He(const Number x) const
  {
      Number ret;
      ret = 0.5*(1. + std::tanh(x/eps));
      return ret;
  }
};

template <int dim, typename LAC>
TwoPhaseFlow<dim,LAC>::TwoPhaseFlow() :
  PDESystemInterface<dim, dim, TwoPhaseFlow<dim,LAC>, LAC>("Two phase flows",
                                                     dim+3,1,
                                                     "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(1)-FE_Q(1)]",
                                                     "u,u,p,c,mu","1,0,1,0")
{}


template <int dim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
TwoPhaseFlow<dim,LAC>::
energies_and_residuals(const typename DoFHandler<dim>::active_cell_iterator &cell,
                       FEValuesCache<dim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool ) const
{
    ResidualType alpha = 0;
    this->reinit(alpha, cell, fe_cache);

    const FEValuesExtractors::Vector velocity(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Scalar concentration(dim+1);
    const FEValuesExtractors::Scalar aux(dim+2);


    auto &us_dot = fe_cache.get_values("solution_dot", "velocity", velocity, alpha);
    auto &us = fe_cache.get_values("solution", "velocity", velocity, alpha);
    auto &grad_us = fe_cache.get_gradients("solution", "velocity", velocity, alpha);
    auto &div_us = fe_cache.get_divergences("solution", "velocity", velocity, alpha);
    auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "velocity", velocity, alpha);

    auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
//    auto &grad_ps = fe_cache.get_gradients("solution", "p", pressure, alpha);

    auto &cs_dot = fe_cache.get_values("solution_dot", "c", concentration, alpha);
    auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);
    auto &grad_cs = fe_cache.get_gradients("solution", "c", concentration, alpha);

    auto &mus = fe_cache.get_values("solution", "mu", aux, alpha);
    auto &grad_mus = fe_cache.get_gradients("solution", "mu", aux, alpha);

    double dd;
    auto &us_expl = fe_cache.get_values("explicit_solution", "velocity", velocity, dd);

    const unsigned int n_q_points = cs.size();

    auto &JxW = fe_cache.get_JxW_values();
    auto &fev = fe_cache.get_current_fe_values();

    Tensor<1,dim> g;
    g[dim-1]=-9.81;

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const ResidualType &c = cs[q];
        const ResidualType &mu = mus[q];

        const Tensor<1,dim,ResidualType> &grad_c = grad_cs[q];
        const Tensor<1,dim,ResidualType> &grad_mu = grad_mus[q];


        const ResidualType &c_dot = cs_dot[q];

        // f = 100*c^2*(1-c)^2
        // f_prime = df/dc
        ResidualType f_prime = 200.*(c-1.)*(c-1.)*c + 200.*(c-1.)*c*c;

//        ResidualType c_half = c-0.5;
//        ResidualType rho = (rho1 - rho2)*He(c_half)+rho2;
//        ResidualType eta = (eta1 - eta2)*He(c_half)+eta2;
        double c_half = SacadoTools::to_double(c)-0.5;
        double rho = (rho1 - rho2)*He(c_half)+rho2;
        double eta = (eta1 - eta2)*He(c_half)+eta2;

        for (unsigned int i=0; i<local_residuals[0].size(); ++i)
          {
            // cahn-hilliard
            auto test_c = fev[concentration].value(i,q);
            auto grad_test_c = fev[concentration].gradient(i,q);

            auto test_mu = fev[aux].value(i,q);
            auto grad_test_mu = fev[aux].gradient(i,q);
            local_residuals[0][i] += (
                                       c_dot*test_c
                                       +
                                       M0*(SacadoTools::scalar_product(grad_mu,grad_test_c))
                                       +
                                       mu*test_mu
                                       -
(3./2.)*eps*(SacadoTools::scalar_product(grad_c,grad_test_mu))
                  -(24./eps)*(c*(1.-c)*(1.-2.*c))*test_mu

                  +SacadoTools::scalar_product(grad_c,us_expl[q])*test_c

                                     )*JxW[q];

            // NS
            auto test_v = fev[velocity].value(i,q);
            auto sym_grad_test_v = fev[velocity].symmetric_gradient(i,q);
            auto div_test_v = fev[velocity].divergence(i,q);

            auto test_p = fev[pressure].value(i,q);
//            auto grad_test_p = fev[pressure].gradient(i,q);

            ResidualType r =
            (
                                    rho*us_dot[q]*test_v
                                    /*+
                                       rho*(SacadoTools::scalar_product((grad_us[q]*us[q]),test_v)) */
                                    +
                                    eta*(scalar_product(sym_grad_us[q],sym_grad_test_v))
                                    -
                                    ps[q]*div_test_v
                                    -
                                    div_us[q]*test_p
                                    -
                                    rho*(g*test_v)
                                    -
                                    SacadoTools::scalar_product((mu*grad_c),test_v)

                                   )*JxW[q];
            local_residuals[0][i] += r;
          }
    }

}

template <int dim, typename LAC>
void TwoPhaseFlow<dim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim, dim, TwoPhaseFlow<dim,LAC>, LAC>::declare_parameters(prm);

  this->add_parameter(prm, &rho1, "rho 1", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &rho2, "rho 2", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta1, "eta 1", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta2, "eta 2", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eps, "eps", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &M0, "M0", "1", Patterns::Double(0.0));

}



#endif
/*! @} */

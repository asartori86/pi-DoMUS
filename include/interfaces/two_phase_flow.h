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


using namespace SacadoTools;

template <int dim, typename LAC>
class TwoPhaseFlow : public PDESystemInterface<dim, dim, TwoPhaseFlow<dim,LAC>, LAC>
{
public:

  ~TwoPhaseFlow() {}

  TwoPhaseFlow();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back();

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
  double rho3;
  double eta1;
  double eta2;
  double eta3;
  double eps;
  double M0;
  double sigma12;
  double sigma13;
  double sigma23;

  double Sigma1;
  double Sigma2;
  double Sigma3;

  double Lambda;


  template <typename Number>
  Number He(const Number x) const
  {
      Number ret;
      ret = 0.5*(1. + std::tanh(x/eps));
      return ret;
  }

  template <typename number>
  double smooth(const double n1, const double n2, const number c) const
  {
    double c_half = to_double(c)-0.5;
    double n = (n1 - n2)*He(c_half)+n2;
    return n;
  }

  template <typename number>
  double smooth(const double n1,
                const double n2,
                const double n3,
                const number c1,
                const number c2) const
  {
    double c1_half = to_double(c1)-0.5;
    double c2_half = to_double(c2)-0.5;
    double n = (n1 - n3)*He(c1_half)+(n2-n3)*He(c2_half)+n3;
    return n;
  }
};

template <int dim, typename LAC>
TwoPhaseFlow<dim,LAC>::TwoPhaseFlow() :
  PDESystemInterface<dim, dim, TwoPhaseFlow<dim,LAC>, LAC>("Three phase flows",
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
    const FEValuesExtractors::Scalar FEc(dim+1);
    const FEValuesExtractors::Scalar aux(dim+2);


    auto &us_dot = fe_cache.get_values("solution_dot", "velocity", velocity, alpha);
    auto &us = fe_cache.get_values("solution", "velocity", velocity, alpha);
    auto &grad_us = fe_cache.get_gradients("solution", "velocity", velocity, alpha);
    auto &div_us = fe_cache.get_divergences("solution", "velocity", velocity, alpha);
    auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "velocity", velocity, alpha);

    auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
//    auto &grad_ps = fe_cache.get_gradients("solution", "p", pressure, alpha);

    auto &cs_dot = fe_cache.get_values("solution_dot", "c", FEc, alpha);
    auto &cs = fe_cache.get_values("solution", "c", FEc, alpha);
    auto &grad_cs = fe_cache.get_gradients("solution", "c", FEc, alpha);

    auto &mus = fe_cache.get_values("solution", "mu", aux, alpha);
    auto &grad_mus = fe_cache.get_gradients("solution", "mu", aux, alpha);

    double dd=0;
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

//        double rho = smooth(rho1, rho2, rho3, c1, c2);
//        double eta = smooth(eta1, eta2, eta3, c1, c2);
        double rho = smooth(rho1, rho2, c);
        double eta = smooth(eta1, eta2, c);

        for (unsigned int i=0; i<local_residuals[0].size(); ++i)
          {
            // cahn-hilliard
            auto test_c = fev[FEc].value(i,q);

            auto grad_test_c = fev[FEc].gradient(i,q);

            auto test_mu = fev[aux].value(i,q);
            auto grad_test_mu = fev[aux].gradient(i,q);

            local_residuals[0][i] += (
                  c_dot*test_c
                  + M0*scalar_product(grad_mu,grad_test_c)
                  + scalar_product(us_expl[q],grad_c)*test_c
                  + mu*test_mu
                  + 3./2.*sigma12*eps*scalar_product(grad_c,grad_test_mu)
                  - 24.*sigma12/eps*c*(1.-c)*(1.-2.*c)*test_mu


//                  - (2.*sigma12*c1*(1.-c1)*(1.-c1))*test_mu1
//                  - 3./4.*Sigma1*eps*scalar_product(grad_c1,grad_test_mu1)

//                  + c2_dot*test_c2
//                  + M0/Sigma2*scalar_product(grad_mu2,grad_test_c2)
//                  + scalar_product(us_expl[q],grad_c2)*test_c2
//                  + mu2*test_mu2
//                  - (2.*sigma12*c1*c1*c2)*test_mu2
//                  - 3./4.*Sigma2*eps*scalar_product(grad_c2,grad_test_mu2)

                    )*JxW[q];

            // NS
            auto test_v = fev[velocity].value(i,q);
            auto sym_grad_test_v = fev[velocity].symmetric_gradient(i,q);
            auto grad_test_v = fev[velocity].gradient(i,q);
            auto div_test_v = fev[velocity].divergence(i,q);

            auto test_p = fev[pressure].value(i,q);
//            auto grad_test_p = fev[pressure].gradient(i,q);

//            double rho_expl = smooth(rho1, rho2, SacadoTools::to_double(c1s_expl[q]));

//            double alpha = this->get_alpha();

            const Tensor<1,dim,ResidualType> nl1 = grad_us[q]*us_expl[q];
            const Tensor<1,dim,ResidualType> nl2 = grad_test_v*us_expl[q];

            ResidualType r =
            (
//                                    std::sqrt(rho)*(std::sqrt(rho)*us[q] - std::sqrt(rho_expl)*us_expl[q])*alpha*test_v
                                    rho*us_dot[q]*test_v
                                    /*+
                                       rho*(SacadoTools::scalar_product((grad_us[q]*us[q]),test_v)) */
                                    +
                                    eta*(scalar_product(sym_grad_us[q],sym_grad_test_v))

                                    +
                                    0.5*rho*scalar_product(nl1,test_v)
                                    -
                                    0.5*rho*scalar_product(nl2,us[q])
                                    -
                                    ps[q]*div_test_v
                                    -
                                    div_us[q]*test_p
                                    -
                                    rho*(g*test_v)
                  - scalar_product((mu*grad_c),test_v)
//                  - scalar_product((mu2*grad_c2),test_v)

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
  this->add_parameter(prm, &rho3, "rho 3", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta1, "eta 1", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta2, "eta 2", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta3, "eta 3", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eps, "eps", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &M0, "M0", "1", Patterns::Double(0.0));
  this->add_parameter(prm, &Lambda, "Lambda", "1", Patterns::Double(0.0));
  this->add_parameter(prm, &sigma12, "sigma12", "1", Patterns::Double(0.0));
  this->add_parameter(prm, &sigma13, "sigma13", "1", Patterns::Double(0.0));
  this->add_parameter(prm, &sigma23, "sigma23", "1", Patterns::Double(0.0));

}

template <int dim, typename LAC>
void TwoPhaseFlow<dim,LAC>::parse_parameters_call_back()
{
  Sigma1 = sigma12 + sigma13 - sigma23;
  Sigma2 = sigma12 + sigma23 - sigma13;
  Sigma3 = sigma13 + sigma23 - sigma12;
}

#endif
/*! @} */

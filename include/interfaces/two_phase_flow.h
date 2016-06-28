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
class ThreePhaseFlow : public PDESystemInterface<dim, dim, ThreePhaseFlow<dim,LAC>, LAC>
{
public:

  ~ThreePhaseFlow() {}

  ThreePhaseFlow();

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

  double smooth(const double n1, const double n2, const double c) const
  {
    double c_half = c-0.5;
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
ThreePhaseFlow<dim,LAC>::ThreePhaseFlow() :
  PDESystemInterface<dim, dim, ThreePhaseFlow<dim,LAC>, LAC>("Three phase flows",
                                                     dim+7,1,
                                                     "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(1)-FE_Q(1)-FE_Q(1)-FE_Q(1)-FE_Q(1)-FE_Q(1)]",
                                                     "u,u,p,c1,c2,c3,mu1,mu2,mu3","1,0,1,1,1,0,0,0")
{}


template <int dim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
ThreePhaseFlow<dim,LAC>::
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
    const FEValuesExtractors::Scalar FEc1(dim+1);
    const FEValuesExtractors::Scalar FEc2(dim+2);
    const FEValuesExtractors::Scalar FEc3(dim+3);
    const FEValuesExtractors::Scalar aux1(dim+4);
    const FEValuesExtractors::Scalar aux2(dim+5);
    const FEValuesExtractors::Scalar aux3(dim+6);


    auto &us_dot = fe_cache.get_values("solution_dot", "velocity", velocity, alpha);
    auto &us = fe_cache.get_values("solution", "velocity", velocity, alpha);
    auto &grad_us = fe_cache.get_gradients("solution", "velocity", velocity, alpha);
    auto &div_us = fe_cache.get_divergences("solution", "velocity", velocity, alpha);
    auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "velocity", velocity, alpha);

    auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
//    auto &grad_ps = fe_cache.get_gradients("solution", "p", pressure, alpha);

    auto &c1s_dot = fe_cache.get_values("solution_dot", "c", FEc1, alpha);
    auto &c1s = fe_cache.get_values("solution", "c", FEc1, alpha);
    auto &grad_c1s = fe_cache.get_gradients("solution", "c", FEc1, alpha);

    auto &c2s_dot = fe_cache.get_values("solution_dot", "c", FEc2, alpha);
    auto &c2s = fe_cache.get_values("solution", "c", FEc2, alpha);
    auto &grad_c2s = fe_cache.get_gradients("solution", "c", FEc2, alpha);

    auto &c3s_dot = fe_cache.get_values("solution_dot", "c", FEc3, alpha);
    auto &c3s = fe_cache.get_values("solution", "c", FEc3, alpha);
    auto &grad_c3s = fe_cache.get_gradients("solution", "c", FEc3, alpha);

    auto &mu1s = fe_cache.get_values("solution", "mu1", aux1, alpha);
    auto &grad_mu1s = fe_cache.get_gradients("solution", "mu1", aux1, alpha);
    auto &mu2s = fe_cache.get_values("solution", "mu2", aux2, alpha);
    auto &grad_mu2s = fe_cache.get_gradients("solution", "mu2", aux2, alpha);
    auto &mu3s = fe_cache.get_values("solution", "mu3", aux3, alpha);
    auto &grad_mu3s = fe_cache.get_gradients("solution", "mu3", aux3, alpha);

    double dd;
    auto &us_expl = fe_cache.get_values("explicit_solution", "velocity", velocity, dd);
    auto &c1s_expl = fe_cache.get_values("explicit_solution", "conc1", FEc1, dd);
    auto &c2s_expl = fe_cache.get_values("explicit_solution", "conc2", FEc2, dd);
    auto &c3s_expl = fe_cache.get_values("explicit_solution", "conc3", FEc3, dd);

    const unsigned int n_q_points = c1s.size();

    auto &JxW = fe_cache.get_JxW_values();
    auto &fev = fe_cache.get_current_fe_values();

    Tensor<1,dim> g;
    g[dim-1]=-9.81;

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const ResidualType &c1 = c1s[q];
        const ResidualType &c2 = c2s[q];
        const ResidualType &c3 = c3s[q];
        const ResidualType &mu1 = mu1s[q];
        const ResidualType &mu2 = mu2s[q];
        const ResidualType &mu3 = mu3s[q];

        const double &c1_e = c1s_expl[q];
        const double &c2_e = c2s_expl[q];
        const double &c3_e = c3s_expl[q];

        const Tensor<1,dim,ResidualType> &grad_c1 = grad_c1s[q];
        const Tensor<1,dim,ResidualType> &grad_c2 = grad_c2s[q];
        const Tensor<1,dim,ResidualType> &grad_c3 = grad_c3s[q];
        const Tensor<1,dim,ResidualType> &grad_mu1 = grad_mu1s[q];
        const Tensor<1,dim,ResidualType> &grad_mu2 = grad_mu2s[q];
        const Tensor<1,dim,ResidualType> &grad_mu3 = grad_mu3s[q];

        const ResidualType &c1_dot = c1s_dot[q];
        const ResidualType &c2_dot = c2s_dot[q];
        const ResidualType &c3_dot = c3s_dot[q];

        double rho = smooth(rho1, rho2, rho3, c1, c2);
        double eta = smooth(eta1, eta2, eta3, c1, c2);

        for (unsigned int i=0; i<local_residuals[0].size(); ++i)
          {
            // cahn-hilliard
            auto test_c1 = fev[FEc1].value(i,q);
            auto test_c2 = fev[FEc2].value(i,q);
            auto test_c3 = fev[FEc3].value(i,q);

            auto grad_test_c1 = fev[FEc1].gradient(i,q);
            auto grad_test_c2 = fev[FEc2].gradient(i,q);
            auto grad_test_c3 = fev[FEc3].gradient(i,q);

            auto test_mu1 = fev[aux1].value(i,q);
            auto test_mu2 = fev[aux2].value(i,q);
            auto test_mu3 = fev[aux3].value(i,q);
            auto grad_test_mu1 = fev[aux1].gradient(i,q);
            auto grad_test_mu2 = fev[aux2].gradient(i,q);
            auto grad_test_mu3 = fev[aux3].gradient(i,q);

            local_residuals[0][i] += (
                  c1_dot*test_c1
                  + M0/Sigma1*scalar_product(grad_mu1,grad_test_c1)
                  + scalar_product(us_expl[q],grad_c1)*test_c1
                  + mu1*test_mu1
                  - (2.*sigma12*c1*c2*c2 + 2.*sigma13*c1*c3*c3
                     + c2*c3*(Sigma1*c1+Sigma2*c2+Sigma3*c3) + c1*c2*c3*Sigma1
                     +2./3.*Lambda*c1*(c2_e*c2_e*c3_e*c3_e
                                       + 0.5*c2*c2*c3_e*c3_e
                                       + 0.5*c2_e*c2_e*c3*c3
                                       + c2*c2*c3*c3))*test_mu1
                  - 3./4.*Sigma1*eps*scalar_product(grad_c1,grad_test_mu1)

                  + c2_dot*test_c2
                  + M0/Sigma2*scalar_product(grad_mu2,grad_test_c2)
                  + scalar_product(us_expl[q],grad_c2)*test_c2
                  + mu2*test_mu2
                  - (2.*sigma12*c1*c1*c2 + 2.*sigma23*c2*c3*c3
                     + c1*c3*(Sigma1*c1+Sigma2*c2+Sigma3*c3) + c1*c2*c3*Sigma2
                     +2./3.*Lambda*c2*(c1_e*c1_e*c3_e*c3_e
                                       + 0.5*c1*c1*c3_e*c3_e
                                       + 0.5*c1_e*c1_e*c3*c3
                                       + c1*c1*c3*c3))*test_mu2
                  - 3./4.*Sigma2*eps*scalar_product(grad_c2,grad_test_mu2)


                  + c3_dot*test_c3
                  + M0/Sigma3*scalar_product(grad_mu3,grad_test_c3)
                  + scalar_product(us_expl[q],grad_c3)*test_c3
                  + mu3*test_mu3
                  - (2.*sigma13*c1*c1*c3 + 2.*sigma23*c2*c2*c3
                     + c1*c2*(Sigma1*c1+Sigma2*c2+Sigma3*c3) + c1*c2*c3*Sigma3
                     +2./3.*Lambda*c3*(c1_e*c1_e*c2_e*c2_e
                                       + 0.5*c1*c1*c2_e*c2_e
                                       + 0.5*c1_e*c1_e*c2*c2
                                       + c1*c1*c2*c2))*test_mu3
                  - 3./4.*Sigma3*eps*scalar_product(grad_c3,grad_test_mu3)
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
                  - scalar_product((mu1*grad_c1),test_v)
                  - scalar_product((mu2*grad_c2),test_v)
                  - scalar_product((mu3*grad_c3),test_v)

                                   )*JxW[q];
            local_residuals[0][i] += r;
          }
    }

}

template <int dim, typename LAC>
void ThreePhaseFlow<dim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim, dim, ThreePhaseFlow<dim,LAC>, LAC>::declare_parameters(prm);

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
void ThreePhaseFlow<dim,LAC>::parse_parameters_call_back()
{
  Sigma1 = sigma12 + sigma13 - sigma23;
  Sigma2 = sigma12 + sigma23 - sigma13;
  Sigma3 = sigma13 + sigma23 - sigma12;
}

#endif
/*! @} */

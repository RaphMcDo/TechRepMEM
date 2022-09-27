#include <TMB.hpp> // Links in the TMB libraries

#include "sim_multinom.hpp"
#include "distmat.hpp"

template<class Type>
Type objective_function<Type>::operator() ()
{

  using namespace density;

  DATA_MATRIX(A); // The halibut data for 1000 hooks
  // DATA_MATRIX(X); // Covariates matrix
  // DATA_MATRIX(Z); //Random effects matrix
  // PARAMETER_VECTOR(betat); // Parameters for lambda t (target species)
  // PARAMETER_VECTOR(betant); // Parameters for lambda nt (non-target species)
  PARAMETER(theta); // Parameters for pnt (probability of caught non-target fish escaping)
  DATA_VECTOR(s); // Soak time
  // PARAMETER_VECTOR(log_H_aniso);
  //
  // matrix<Type> H_aniso(2,2);
  // H_aniso(0,0) = exp(log_H_aniso(0));
  // H_aniso(1,0) = log_H_aniso(1);
  // H_aniso(0,1) = log_H_aniso(1);
  // H_aniso(1,1) = (1+log_H_aniso(1)*log_H_aniso(1)) / exp(log_H_aniso(0));
  // REPORT(H_aniso);

  DATA_IVECTOR(n);
  DATA_INTEGER(n_k);
  DATA_INTEGER(n_t);

  PARAMETER_VECTOR(rand_t); //Random effect in each year for target
  PARAMETER_VECTOR(rand_nt); //Random effect in each year for non-target
  PARAMETER_VECTOR(omegat); // The random field for lambda t
  PARAMETER_VECTOR(omegant); // The random field for lambda nt

  // DATA_MATRIX(D); // Distance matrix
  DATA_MATRIX(locations);
  PARAMETER(lognut);
  PARAMETER(lognunt);
  PARAMETER_VECTOR(logPhit);
  PARAMETER_VECTOR(logPhint);
  PARAMETER(logSigmat);
  PARAMETER(logSigmant);
  PARAMETER(log_sigma_rand_t);
  PARAMETER(log_sigma_rand_nt);
  PARAMETER(mean_ar_t);
  PARAMETER(mean_ar_nt);
  PARAMETER(logitphi_ar_t);
  PARAMETER(logitphi_ar_nt);
  // spatial parameters
  Type nut=exp(lognut); // smoothness parameter
  Type nunt=exp(lognunt);
  vector <Type> phit=exp(logPhit); // range parameter
  vector <Type> phint=exp(logPhint);
  Type sigt=exp(logSigmat); // variance
  Type signt=exp(logSigmant);
  Type sigma_rand_t = exp(log_sigma_rand_t);
  Type sigma_rand_nt = exp(log_sigma_rand_nt);
  Type phi_ar_t = 2*invlogit(logitphi_ar_t)-1;
  Type phi_ar_nt = 2*invlogit(logitphi_ar_nt)-1;

  vector <Type> nll_joint(5); nll_joint.setZero();

  // Covaraince matirx for random field for lambda t
  int counter = 0;
  for (int t=0; t < n_t; t++){

    if (t > 0) counter = counter + n(t-1);

    matrix <Type> temp_loc(n(t),2);
    for (int i = 0; i < n(t); i++){
      temp_loc(i,0) = locations((i+counter),0);
      temp_loc(i,1) = locations((i+counter),1);
    }

    matrix <Type> D(n(t),n(t));
    // D = DistMat2d_aniso(temp_loc,H_aniso);
    D = DistMat2d(temp_loc);

    matrix<Type> St(n(t),n(t));
    St.setZero();
    for(int i=0; i<n(t); ++i){
      St(i,i) = sigt*sigt;
    }
    for(int i=0; i<n(t); ++i){
      for(int j=i+1; j<n(t); ++j){
        St(i,j) = sigt*sigt*matern(D(i,j), phit(t), nut); //Matern Covariance Function
        St(j,i) = St(i,j);
      }
    }

    // Covaraince matirx for random field for lambda nt
    matrix<Type> Snt(n(t),n(t));
    Snt.setZero();
    for(int i=0; i<n(t); ++i){
      Snt(i,i) = signt*signt;
    }
    for(int i=0; i<n(t); ++i){
      for(int j=i+1; j<n(t); ++j){
        Snt(i,j) = signt*signt*matern(D(i,j), phint(t), nunt); //Matern Covariance Function
        Snt(j,i) = Snt(i,j);
      }
    }

    vector <Type> temp_omegat(n(t));
    vector <Type> temp_omegant(n(t));

    for (int i = 0; i < n(t); i++) {
      temp_omegat(i) = omegat((counter+i)) ;
      temp_omegant(i) = omegant((counter+i));
    }

    nll_joint(0) += density::MVNORM(St)(temp_omegat);
    nll_joint(1) += density::MVNORM(Snt)(temp_omegant);

  }

  //AR1 on temporal random effect
  for (int t = 1; t < n_t; t++) {
    nll_joint(3) -= dnorm(rand_t(t), mean_ar_t + phi_ar_t*rand_t(t-1), sigma_rand_t, true);
    nll_joint(4) -= dnorm(rand_nt(t), mean_ar_nt + phi_ar_nt*rand_nt(t-1), sigma_rand_nt, true);
  }

  // ldat - the relative abundance indices for target species (halibut)
  vector <Type> ldat(A.col(0).size());
  vector<Type> ldant(A.col(0).size());
  int counter1 = 0;
  for (int t=0; t < n_t; t++){
    if (t > 0) counter1 = counter1+ n(t-1);
    for(int i=0; i<n(t); ++i){
      // Adding random field and covariates to lambda t
      ldat((counter1+i))=exp(rand_t(t)+omegat((counter1+i)));
    }

    // ldant - the relative abundance indices for non-target species
    for(int i=0; i<n(t); ++i){
      // Adding omega (random field) and covariates to lambda nt
      ldant((counter1+i))=exp(rand_nt(t)+omegant((counter1+i)));
    }
  }

  // pnt - the probability of caught non-target species escaping
  Type pnt;
  pnt=invlogit(theta);
  ADREPORT(pnt);

  // Corresponding probabilities for N_t, N_nt and N_b+e for 700 hooks
  // N_t is the number of target species caught
  // N_nt is the number of non-target species caught
  // N_b+e is the number of empty hooks (baited + unbaited)

  int counter2 = 0;
  matrix<Type> q(A.col(0).size(), n_k);
  for (int t = 0; t < n_t; t++){
    if (t > 0) counter2 = counter2 + n(t-1);
    for(int i=0;i<n(t);i++){
      q((counter2+i),0)=(Type(1)-exp(-(ldat((counter2+i))+ldant((counter2+i)))*s((counter2+i))))*(ldat((counter2+i))/(ldat((counter2+i))+ldant((counter2+i))));
      q((counter2+i),1)=(Type(1)-exp(-(ldat((counter2+i))+ldant((counter2+i)))*s((counter2+i))))*(ldant((counter2+i))/(ldat((counter2+i))+ldant((counter2+i))))*(Type(1)-pnt);
      q((counter2+i),2)=1-q((counter2+i),0)-q((counter2+i),1);
    }
  }

  // The likelihood function for multinomial distribution
  int counter3 = 0;
  for (int t = 0; t < n_t; t++){
    if (t > 0) counter3 = counter3+n(t-1);
    for(int i=0; i < n(t); i++){
      // vector<Type> p_row=p.row(i);
      // vector<Type> H_row=H.row(i);
      vector<Type> q_row=q.row((counter3+i));
      vector<Type> A_row=A.row((counter3+i));
      // nll_joint(2) -= dmultinom(H_row,p_row,true);
      nll_joint(2) -= dmultinom(A_row,q_row,true);
    }
  }

  REPORT(nll_joint);

  // REPORT(D);
  REPORT(ldat);
  // ADREPORT(ldat)
  REPORT(ldant);
  // ADREPORT(ldant);
  REPORT(pnt);
  REPORT(q);
  // REPORT(betat);
  // REPORT(betant);
  REPORT(rand_t);
  REPORT(rand_nt);
  ADREPORT(rand_t);
  ADREPORT(rand_nt);
  REPORT(omegat);
  REPORT(omegant);
  ADREPORT(phit);
  ADREPORT(phint);
  Type vart = sigt*sigt;
  Type varnt = signt*signt;
  ADREPORT(vart);
  ADREPORT(varnt);
  REPORT(sigma_rand_t);
  ADREPORT(sigma_rand_t);
  REPORT(sigma_rand_nt);
  ADREPORT(sigma_rand_nt);
  REPORT(mean_ar_t);
  ADREPORT(mean_ar_t);
  REPORT(mean_ar_nt);
  ADREPORT(mean_ar_nt);
  REPORT(phi_ar_t);
  ADREPORT(phi_ar_t);
  REPORT(phi_ar_nt);
  ADREPORT(phi_ar_nt);

  Type nll = nll_joint.sum();
  return nll;
}

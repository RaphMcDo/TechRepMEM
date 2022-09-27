#include <TMB.hpp> // Links in the TMB libraries

#include "sim_multinom.hpp"
#include "distmat.hpp"
#include "covariance.hpp"
#include "kriging.hpp"

template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}

template<class Type>
Type objective_function<Type>::operator() ()
{

  using namespace density;

  DATA_INTEGER(n_t); //Number of years
  DATA_IVECTOR(n); //For A
  DATA_IVECTOR(krig_n);

  PARAMETER_VECTOR(omegat); // The random field for lambda t
  PARAMETER_VECTOR(omegant); // The random field for lambda nt
  PARAMETER_MATRIX(krig_field);


  // DATA_MATRIX(D); // Distance matrix
  DATA_MATRIX(locations);
  DATA_MATRIX(krig_locations);
  PARAMETER(lognut);
  PARAMETER(lognunt);
  PARAMETER(logPhit);
  PARAMETER(logPhint);
  PARAMETER(logSigmat);
  PARAMETER(logSigmant);
  PARAMETER(mean_par);
  // spatial parameters
  Type nut=exp(lognut); // smoothness parameter
  Type nunt=exp(lognunt);
  // vector <Type> phit=exp(logPhit); // range parameter
  // vector <Type> phint=exp(logPhint);
  Type phit=exp(logPhit);
  Type phint=exp(logPhint);
  Type sigt=exp(logSigmat); // variance
  Type signt=exp(logSigmant);

  vector <Type> nll_joint(2); nll_joint.setZero();

  // Covaraince matirx for random field for lambda t
  int counter = 0;
  int krig_count = 0;
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
        St(i,j) = sigt*sigt*matern(D(i,j), phit, nut); //Matern Covariance Function
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
        Snt(i,j) = signt*signt*matern(D(i,j), phint, nunt); //Matern Covariance Function
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
    nll_joint(0) += density::MVNORM(Snt)(temp_omegant);

    // This section is for interpolating the random field.
    if (krig_n(t)>0){

      if (t > 0) krig_count = krig_count+krig_n(t-1);

      covariance<Type> cov1(sigt, phit, nut, 2);
      covariance<Type> cov2(signt, phint, nunt, 2);

      matrix <Type> temp_krig_loc(krig_n(t),2);
      for (int i = 0; i < krig_n(t); i++){
        temp_krig_loc(i,0) = krig_locations((i+krig_count),0);
        temp_krig_loc(i,1) = krig_locations((i+krig_count),1);
      }

      matrix <Type> B = DistMat2D_2var(temp_krig_loc,temp_loc);

      vector<Type> mean_vector(n(t));
      for (int i=0; i < n(t); i++){
        mean_vector(i)=mean_par;
      }
      matrix<Type> cross_covariances1 = cov1(B);
      kriging<Type> krig(St, // Covariance matrix for stations
                         mean_vector, // Mean vector for stations
                         temp_omegat); // Random effects for stations

      vector<Type> krig_out(2);
      for( int i=0; i< krig_n(t); i++) {
        krig_out = krig(cross_covariances1.row(i), // Cross-covariance matrix
                        0, // Marginal mean
                        sigt); // Marginal standard deviation
        nll_joint(1) -= dnorm(krig_field((i+krig_count),0),
                  krig_out(0),
                  krig_out(1),
                  true);
      }

      matrix<Type> cross_covariances2 = cov2(B);
      kriging<Type> krig2(Snt, // Covariance matrix for stations
                          mean_vector, // Mean vector for stations
                          temp_omegant); // Random effects for stations

      vector<Type> krig_out2(2);
      for( int i=0; i< krig_n(t); i++) {
        krig_out2 = krig2(cross_covariances2.row(i), // Cross-covariance matrix
                          0, // Marginal mean
                          signt); // Marginal standard deviation
        nll_joint(1) -= dnorm(krig_field((i+krig_count),1),
                  krig_out2(0),
                  krig_out2(1),
                  true);
      }

    }

  }
  REPORT(krig_field);

  Type nll=nll_joint.sum();
  return nll;
}

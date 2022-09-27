#include <TMB.hpp> // Links in the TMB libraries

#include "sim_multinom.hpp"
#include "distmat.hpp"

template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}

template<class Type>
Type objective_function<Type>::operator() ()
{

  using namespace density;

  DATA_MATRIX(H); //Halibut data with fine_scale info on hook condition
  DATA_MATRIX(A); //Halibut data with non-fine scale info on hook condition
  PARAMETER(theta); // Parameters for pnt (probability of caught non-target fish escaping)
  DATA_VECTOR(s); // Soak time

  DATA_INTEGER(n_t); //Number of years
  DATA_IVECTOR(n); //For A
  DATA_INTEGER(n_k); //For A
  DATA_IVECTOR(n2); //For H
  DATA_INTEGER(n_k2); //For H
  DATA_IVECTOR(H_i); //Indicator if subset of data present for full MEM to identify which ldat/ldant to be related to
  DATA_IVECTOR(H_j); //Indicator if subset of data present for multinomial part
  DATA_IVECTOR(vess_id);

  PARAMETER_VECTOR(rand_t); //Random effect in each year for target
  PARAMETER_VECTOR(rand_nt); //Random effect in each year for non-target
  PARAMETER_VECTOR(omegat); // The random field for lambda t
  PARAMETER_VECTOR(omegant); // The random field for lambda nt
  PARAMETER_VECTOR(vess_eff_t);
  PARAMETER_VECTOR(vess_eff_nt);


  DATA_MATRIX(locations);
  PARAMETER(lognut);
  PARAMETER(lognunt);
  PARAMETER(logPhit);
  PARAMETER(logPhint);
  PARAMETER(logSigmat);
  PARAMETER(logSigmant);
  PARAMETER(log_sigma_vess_t);
  PARAMETER(log_sigma_vess_nt);
  PARAMETER(log_sigma_rand_t);
  PARAMETER(log_sigma_rand_nt);
  PARAMETER(mean_rand_t);
  PARAMETER(mean_rand_nt);
  // spatial parameters
  Type nut=exp(lognut); // smoothness parameter
  Type nunt=exp(lognunt);
  Type phit=exp(logPhit);
  Type phint=exp(logPhint);
  Type sigt=exp(logSigmat); // variance
  Type signt=exp(logSigmant);
  Type sigma_vess_t = exp(log_sigma_vess_t);
  Type sigma_vess_nt = exp(log_sigma_vess_nt);
  Type sigma_rand_t = exp(log_sigma_rand_t);
  Type sigma_rand_nt = exp(log_sigma_rand_nt);

  vector <Type> nll_joint(7); nll_joint.setZero();

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
    nll_joint(1) += density::MVNORM(Snt)(temp_omegant);

  }

  SIMULATE{
    int counter_sim = 0;
    for (int t=0; t < n_t; t++){

      if (t > 0) counter_sim = counter_sim + n(t-1);

      matrix <Type> temp_loc(n(t),2);
      for (int i = 0; i < n(t); i++){
        temp_loc(i,0) = locations((i+counter_sim),0);
        temp_loc(i,1) = locations((i+counter_sim),1);
      }

      matrix <Type> D(n(t),n(t));
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

      density::MVNORM(St).simulate(temp_omegat);
      density::MVNORM(Snt).simulate(temp_omegant);

      for (int i = 0; i < n(t); i++) {
        omegat((counter_sim+i)) = temp_omegat(i);
        omegant((counter_sim+i)) = temp_omegant(i);
      }

    }
    REPORT(omegat);
    REPORT(omegant);
  }

  //Simulate soaking time
  SIMULATE{
    int county = 0;
    for (int t=0; t < n_t; t++){
      if (t>0)  county = county + n(t-1);
      for (int i = 0; i < n(t); i++){
        s((county+i)) = exp(rnorm(log(Type(450)),Type(0.2)));
      }
    }
    REPORT(s);
  }

  Type n_vess = vess_eff_t.size();
  for (int i=0; i < n_vess; i++){
    nll_joint(3) -= dnorm(vess_eff_t(i),Type(0.0),sigma_vess_t,true);
    nll_joint(4) -= dnorm(vess_eff_nt(i),Type(0.0),sigma_vess_nt,true);
  }


  nll_joint(5) -= dnorm(rand_t(0),mean_rand_t,sigma_rand_t,true);
  nll_joint(6) -= dnorm(rand_nt(0),mean_rand_nt,sigma_rand_nt,true);
  for (int t=1; t <n_t; t++){
    nll_joint(5) -= dnorm(rand_t(t),rand_t(t-1),sigma_rand_t,true);
    nll_joint(6) -= dnorm(rand_nt(t),rand_nt(t-1),sigma_rand_nt,true);
  }

  SIMULATE{
    for (int i=0; i < n_vess; i++){
      vess_eff_t(i) = rnorm(Type(0.0),sigma_vess_t);
      vess_eff_nt(i) = rnorm(Type(0.0),sigma_vess_nt);
    }
    REPORT(vess_eff_t);
    REPORT(vess_eff_nt);

    rand_t(0) = mean_rand_t+rnorm(Type(0.0),sigma_rand_t);
    rand_nt(0) = mean_rand_nt+rnorm(Type(0.0),sigma_rand_nt);
    for (int t=1; t<n_t; t++){
      rand_t(t) = rand_t(t-1)+rnorm(Type(0.0),sigma_rand_t);
      rand_nt(t) = rand_nt(t-1)+rnorm(Type(0.0),sigma_rand_nt);
    }
    REPORT(rand_t);
    REPORT(rand_nt);
  }

  // ldat - the relative abundance indices for target species (halibut)
  vector <Type> ldat(A.col(0).size());
  vector<Type> ldant(A.col(0).size());
  int counter1 = 0;
  for (int t=0; t < n_t; t++){
    if (t > 0) counter1 = counter1+ n(t-1);
    for(int i=0; i<n(t); ++i){
        ldat((counter1+i))=exp(rand_t(t)+vess_eff_t(vess_id((counter1+i)))+omegat((counter1+i)));
    }

    // ldant - the relative abundance indices for non-target species
    for(int i=0; i<n(t); ++i){
      // Adding omega (random field) and covariates to lambda nt
      ldant((counter1+i))=exp(rand_nt(t)+vess_eff_nt(vess_id((counter1+i)))+omegant((counter1+i)));
    }
  }

  // pnt - the probability of caught non-target species escaping
  Type pnt;
  pnt=invlogit(theta);
  ADREPORT(pnt);

  //Reduced MEM

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

  // Corresponding probabilities for N_b, N_t, N_nt and N_e for 300 hooks
  // N_b is the nmuber of baited hooks
  // N_t is the number of target species caught
  // N_nt is the number of non-target species caught
  // N_e is the number of empty hooks
  //Fine-scale hook info (full MEM)

  int counter4 = 0;
  matrix<Type> p(H.col(0).size(), n_k2);
  for (int t = 0; t < n_t; t++){
    if (t > 0) counter4 = counter4 + n2(t-1);
    for(int i=0;i<n2(t);i++){
      p((counter4+i),0)=exp(-(ldat(H_i(counter4+i))+ldant(H_i(counter4+i)))*s(H_i(counter4+i)));
      p((counter4+i),1)=(Type(1)-exp(-(ldat(H_i(counter4+i))+ldant(H_i(counter4+i)))*s(H_i(counter4+i))))*(ldat(H_i(counter4+i))/(ldat(H_i(counter4+i))+ldant(H_i(counter4+i))));
      p((counter4+i),2)=(Type(1)-exp(-(ldat(H_i(counter4+i))+ldant(H_i(counter4+i)))*s(H_i(counter4+i))))*(ldant(H_i(counter4+i))/(ldat(H_i(counter4+i))+ldant(H_i(counter4+i))))*(Type(1)-pnt);
      p((counter4+i),3)=1-p((counter4+i),0)-p((counter4+i),1)-p((counter4+i),2);
    }
  }

  // The likelihood function for multinomial distribution
  int counter3 = 0;
  int counter_H = 0;
  for (int t = 0; t < n_t; t++){
    if (t > 0) counter3 = counter3+n(t-1);
    for(int i=0; i < n(t); i++){
      if (H_j(counter3+i)==0){
        vector<Type> q_row=q.row((counter3+i));
        vector<Type> A_row=A.row((counter3+i));
        nll_joint(2) -= dmultinom(A_row,q_row,true);
      } else if (H_j(counter3+i)==1){
        vector<Type> q_row=q.row((counter3+i));
        vector<Type> A_row=A.row((counter3+i));
        vector<Type> p_row=p.row((counter_H));
        vector<Type> H_row=H.row((counter_H));
        nll_joint(2) -= (dmultinom(A_row,q_row,true) + dmultinom(H_row,p_row,true));
        counter_H += 1;
      }
    }
  }

  SIMULATE{
    DATA_VECTOR(hook_num);
    int counter_sim3 = 0;
    int counter_simH = 0;
    for (int t = 0; t < n_t; t++){
      if (t > 0) counter_sim3 = counter_sim3+n(t-1);
      for(int i=0; i < n(t); i++){
        if (H_j(counter_sim3+i)==0){
          vector<Type> q_row=q.row((counter_sim3+i));
          vector<Type> A_row=A.row((counter_sim3+i));
          A_row = sim_multinom(hook_num(0)+hook_num(1),q_row);
          A.row((counter_sim3+i))=A_row;
        } else if (H_j(counter_sim3+i)==1){
          vector<Type> q_row=q.row((counter_sim3+i));
          vector<Type> A_row=A.row((counter_sim3+i));
          vector<Type> p_row=p.row((counter_simH));
          vector<Type> H_row=H.row((counter_simH));
          A_row = sim_multinom(hook_num(0),q_row);
          H_row = sim_multinom(hook_num(1),p_row);
          A.row((counter_sim3+i))=A_row;
          H.row(counter_simH)=H_row;
          counter_simH += 1;
        }
      }
    }
    REPORT(H);
    REPORT(A);
  }

  REPORT(nll_joint);

  REPORT(ldat);
  REPORT(ldant);
  REPORT(pnt);
  REPORT(q);
  REPORT(p);
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
  ADREPORT(ldat);
  ADREPORT(ldant);
  REPORT(vess_eff_t);
  ADREPORT(vess_eff_t);
  REPORT(vess_eff_nt);
  ADREPORT(vess_eff_nt);
  ADREPORT(sigma_vess_t);
  ADREPORT(sigma_vess_nt);
  ADREPORT(sigma_rand_t);
  ADREPORT(sigma_rand_nt);
  ADREPORT(mean_rand_t);
  ADREPORT(mean_rand_nt);

  Type nll = nll_joint.sum();
  return nll;
}

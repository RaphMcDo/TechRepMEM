template<class Type>
class kriging {
  private:
    matrix<Type> predictor_covariance;
    matrix<Type> predictor_precision;

    vector<Type> predictor_means;
    vector<Type> predictor_vals;
  public:
    kriging(matrix<Type> predictor_covariance,
            vector<Type> predictor_means,
            vector<Type> predictor_vals);
    kriging() = default;

    vector<Type> operator() (vector<Type> cross_covariance,
                             Type marginal_mean,
                             Type marginal_std_dev);
};



namespace krig_funs {
  template<class Type> matrix<Type> get_inv(matrix<Type> mat);
}

template<class Type>
kriging<Type>::kriging(matrix<Type> predictor_covariance,
                       vector<Type> predictor_means,
                       vector<Type> predictor_vals) :
  predictor_covariance(predictor_covariance),
  predictor_means(predictor_means),
  predictor_vals(predictor_vals) {
  this->predictor_precision = krig_funs::get_inv(this->predictor_covariance);
}

template<class Type>
vector<Type> kriging<Type>::operator() (vector<Type> cross_covariance,
                                        Type marginal_mean,
                                        Type marginal_std_dev) {
  vector<Type> ans(2); // 0 is conditional mean, 1 is conditional std. dev.
  ans(0) = marginal_mean + (
      cross_covariance.matrix().transpose() * predictor_precision *
      (predictor_vals - predictor_means).matrix()
    )(0,0);
  ans(1) =  sqrt(pow(marginal_std_dev,2) - (
    cross_covariance.matrix().transpose() * predictor_precision * cross_covariance.matrix()
  )(0,0));

  return ans;
}


template<class Type>
matrix<Type> krig_funs::get_inv(matrix<Type> mat)  {
  matrix<Type> ans(0,0);
  if( mat.rows() > 0 ) {
    ans.resizeLike(mat);
    ans = atomic::matinv(mat);
  } else {}
  return ans;
}

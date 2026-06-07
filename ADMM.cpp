#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
//using namespace std;

//The gradient of the missing probability model
//[[Rcpp::export]]
arma::mat gradCPP(arma::mat Z, arma::vec R, arma::vec betaw){
  int n = Z.n_rows, p = Z.n_cols;
  arma::vec Zbeta = arma::zeros(n), mu = arma::zeros(n), grad = arma::zeros(p);
  Zbeta = Z*betaw;
  mu = exp(Zbeta)/(exp(Zbeta)+1);
  grad = Z.t()*(R-mu)/n;
  return grad;
}

//The Hessian matrix of the missing probability model  
//[[Rcpp::export]]
arma::mat hessCPP(arma::mat Z, arma::vec R, arma::vec betaw){
  Environment base("package:base");
  Function sweep = base["sweep"];
  int n = Z.n_rows, p = Z.n_cols;
  arma::vec Zbeta = arma::zeros(n), sigma = arma::zeros(n);
  arma::mat WZ = arma::zeros(n, p), hess = arma::zeros(p, p);
  Zbeta = Z*betaw;
  sigma = (exp(Zbeta)/(exp(Zbeta)+1))%(1/(exp(Zbeta)+1));
  WZ = as<arma::mat>(sweep(Z, 1, sigma,"*"));
  hess = Z.t()*WZ/n;
  return hess;
}

//The parallel missing probability estimation algorithm 
//[[Rcpp::export]]
arma::mat paraBetampCPP(arma::mat Z, arma::vec R, int K, double eps, int maxstep){
  
  int n = Z.n_rows, p = Z.n_cols, nk = n/K, iteration = 0;
  arma::vec final = arma::zeros(p+2);
  double distance = 1, time_iter = 0;
  arma::vec betaini = arma::zeros(p), beta = arma::zeros(p);
  while((distance>eps)&&(iteration<maxstep)){
    
    arma::mat hessmpk = arma::zeros(p, p), gradmpk = arma::zeros(p), Zk = arma::mat(nk, p);
    arma::vec Rk = arma::zeros(nk), timek = arma::zeros(K);
    
    for(int k = 0; k < K; k++){
      Zk = Z.rows(nk*k, nk*(k+1)-1);
      Rk = R.rows(nk*k, nk*(k+1)-1);
      auto start_timek = std::chrono::high_resolution_clock::now();
      hessmpk = hessmpk+hessCPP(Zk, Rk, betaini);
      gradmpk = gradmpk+gradCPP(Zk, Rk, betaini);
      auto finish_timek = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_timek = finish_timek - start_timek;
      timek(k) = elapsed_timek.count();
    }
    time_iter = time_iter+timek(timek.index_max());
    
    auto start_iter = std::chrono::high_resolution_clock::now();
    beta = betaini+inv(hessmpk)*gradmpk;
    auto finish_iter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_iter = finish_iter - start_iter;
    time_iter = time_iter+elapsed_iter.count();
    
    distance = accu(abs(beta-betaini));
    betaini = beta;
    iteration = iteration+1;
  }
  
  final.subvec(0, p-1) = beta;
  final(p) = time_iter;
  final(p+1) = iteration;
  return final;
  
}

//The soft-thresholding solution of r
arma::vec shrinkcpp(arma::vec u, arma::vec v){
  arma::vec w = (1+sign(u-v))/2%(u-v)-(1+sign(-u-v))/2%(-u-v);
  return w;
}

//The check loss function
double loss(arma::vec u, double tau){
  int n = u.size();
  arma::vec loss = arma::zeros(n);
  for(int i = 0; i < n; i++){
    if(u(i)>0){
      loss(i) = tau*u(i);
    }
    else{
      loss(i) = (tau-1)*u(i);
    }
  }
  return accu(loss);
}

//[[Rcpp::export]]
//The parallel QR-ADMM algorithm
arma::vec paraQRCPP(arma::mat x, arma::vec y, arma::vec numk, int K, double tau, double rho, double eps, int maxstep, bool intercept){
  
  int n = x.n_rows;
  if(intercept){
    x.insert_cols(0, arma::ones(n));
  }
  int p = x.n_cols;
  arma::vec betaini = arma::zeros(p), vini = arma::zeros(n); 
  arma::vec beta = arma::zeros(p), r = arma::zeros(n), v = arma::zeros(n); 
  arma::mat betakini = arma::zeros(p,K), betak = arma::zeros(p,K), ukini = arma::zeros(p,K), uk = arma::zeros(p,K);
  arma::vec xbeta = arma::zeros(n);
  double time = 0, max_prep = 0, time_reduce = 0, max_map = 0; 
  arma::vec time_map = arma::zeros(K);
  arma::vec final = arma::zeros(p+2);
  
  arma::cube dat = arma::zeros<arma::cube>(p,p,K);
  int counter = 0;
  for(int k = 0; k < K; k++){
    arma::mat tmp = arma::zeros(p,p), xk = x.rows(counter, counter+numk(k)-1);
    auto start_prep = std::chrono::high_resolution_clock::now();
    if(numk(k) > p) 
      tmp = inv(xk.t()*xk+arma::eye(p,p));
    else tmp = arma::eye(p,p)-xk.t()*inv(xk*xk.t()+arma::eye(numk(k),numk(k)))*xk;
    auto finish_prep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_prep = finish_prep - start_prep;
    if(elapsed_prep.count() > max_prep) max_prep = elapsed_prep.count();
    dat.slice(k) = tmp;
    counter = counter+numk(k);
  }
  time = time+max_prep;
  
  int iteration = 0;
  double distance = 1;
  
  while((distance>eps|distance==0)&&(iteration<maxstep)){
    
    //update beta
    auto start_reduce = std::chrono::high_resolution_clock::now();
    beta = mean(betakini,1)+mean(ukini,1)/rho;
    auto finish_reduce = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_reduce = finish_reduce - start_reduce;
    time_reduce = elapsed_reduce.count();
    time = time+time_reduce;
    counter = 0;
    
    //update r, betak, uk, v
    for(int k = 0; k < K; k++){
      
      arma::vec yk = y.subvec(counter, counter+numk(k)-1), vk = vini.subvec(counter, counter+numk(k)-1);
      arma::mat xk = x.rows(counter, counter+numk(k)-1);
      
      auto start_map = std::chrono::high_resolution_clock::now();
      //update r
      r.subvec(counter, counter+numk(k)-1) = shrinkcpp(vk/rho+yk-xbeta.subvec(counter, counter+numk(k)-1)-0.5*(2*tau-1)/rho, 0.5*arma::ones<arma::vec>(numk(k))/rho);
      //update betak
      betak.col(k) = dat.slice(k)*(xk.t()*(yk-r.subvec(counter, counter+numk(k)-1)+vk/rho)+beta-ukini.col(k)/rho);
      //update uk and v
      xbeta.subvec(counter, counter+numk(k)-1) = xk*betak.col(k);
      uk.col(k) = ukini.col(k)+rho*(betak.col(k)-beta);
      v.subvec(counter, counter+numk(k)-1) = vini.subvec(counter, counter+numk(k)-1)+rho*(yk-xbeta.subvec(counter, counter+numk(k)-1)-r.subvec(counter, counter+numk(k)-1));
      auto finish_map = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_map = finish_map - start_map;
      time_map(k) = elapsed_map.count();
      counter = counter+numk(k);
      
    }
    max_map = time_map(time_map.index_max());
    time = time+max_map;
    distance = accu(abs(beta-betaini));
    betaini = beta, betakini = betak, ukini = uk, vini = v;
    iteration = iteration + 1;
    
  }
  
  final.subvec(0,p-1) = beta;
  final(p) = time;
  final(p+1) = iteration;
  return final;
  
}

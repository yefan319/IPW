# include <RcppArmadillo.h>
# include <cmath>
//# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double mad(const arma::vec& x) {
  return 1.482602 * arma::median(arma::abs(x - arma::median(x)));
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx1, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - mx(i)) * sx1(i);
  }
  return X;
}

// Asymmetric huber regression adjusted to quantile tau for initialization 
// [[Rcpp::export]]
void updateHuber(const arma::mat& Z, const arma::vec& res, const double tau, arma::vec& der, arma::vec& grad, const int n, const double rob, const double n1) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur > rob) {
      der(i) = -2 * tau * rob;
    } else if (cur > 0) {
      der(i) = -2 * tau * cur;
    } else if (cur > -rob) {
      der(i) = 2 * (tau - 1) * cur;
    } else {
      der(i) = 2 * (1 - tau) * rob;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateGauss(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const double tau, const double n1, const double h1) {
  der = arma::normcdf(-res * h1) - tau;
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void hessGauss(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::mat& hess, const double tau, const double n1, const double h1) {
  Environment base("package:base");
  Function sweep = base["sweep"];
  der = arma::normpdf(res * h1) * h1;
  arma::mat WZ = as<arma::mat>(sweep(Z, 1, der,"*"));
  hess = n1 * Z.t() * WZ;
}

// [[Rcpp::export]]
arma::vec Loggrad(arma::mat X, arma::vec y, arma::vec beta){
  int n = X.n_rows;
  arma::vec Xb = X*beta;
  arma::vec mu = exp(Xb)/(exp(Xb)+1);
  arma::vec grad = X.t()*(y-mu)/n;
  return(grad);
}

// [[Rcpp::export]]
arma::mat Loghess(arma::mat X, arma::vec beta){
  Environment base("package:base");
  Function sweep = base["sweep"];
  int n = X.n_rows;
  arma::vec Xb = X*beta;
  arma::vec mu = exp(Xb)/(exp(Xb)+1);
  arma::vec sigma = mu%(1-mu);
  arma::mat XW = as<arma::mat>(sweep(X, 1, sigma, "*"));
  arma::mat hess = X.t()*XW/n;
  return(hess);
}

// Functions with an upper bound for the GD step size.
// [[Rcpp::export]]
arma::vec huberReg(const arma::mat& Z, const arma::vec& Y, const double tau, arma::vec& der, arma::vec& gradOld, arma::vec& gradNew, const int n, const int p, 
                   const double n1, const double tol = 0.0001, const double constTau = 1.345, const int iteMax = 5000, const double stepMax = 100.0) {
  double rob = constTau * mad(Y);
  updateHuber(Z, Y, tau, der, gradOld, n, rob, n1);
  arma::vec beta = -gradOld, betaDiff = -gradOld;
  arma::vec res = Y - Z * beta;
  rob = constTau * mad(res);
  updateHuber(Z, res, tau, der, gradNew, n, rob, n1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), stepMax);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    rob = constTau * mad(res);
    updateHuber(Z, res, tau, der, gradNew, n, rob, n1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return beta;
}

// [[Rcpp::export]]
arma::vec Newtoncpp(arma::mat X, arma::vec y, arma::vec betaini, double eps = 0.000001, int maxstep = 100){
  double dis = 2*eps;
  int iteration = 0;
  int p = betaini.size();
  arma::vec beta = arma::zeros(p);
  while((dis > eps)&&(iteration < maxstep)){
    beta = betaini+inv(Loghess(X,betaini))*Loggrad(X,y,betaini);
    dis = sum(abs(beta-betaini));
    betaini = beta;
    iteration = iteration+1;
  }
  return(beta);
}

// [[Rcpp::export]]
arma::vec seqNewtoncpp(arma::mat X, arma::vec y, arma::vec betaini, arma::vec sizeM, double eps = 0.000001, int maxstep = 100){
  
  int p = betaini.size();
  int M = sizeM.size();
  int record = 0;
  arma::mat betarecord = arma::zeros(p, M);
  arma::mat X1 = X.rows(0, sizeM(0)-1);
  arma::vec y1 = y.subvec(0, sizeM(0)-1);
  betarecord.col(0) = Newtoncpp(X1, y1, betaini, eps, maxstep);  
  record = record+sizeM(0);
  arma::mat hessrecord = Loghess(X1, betarecord.col(0));
  for(int m = 1; m < M; m++){
    arma::mat Xm = X.rows(record, record+sizeM(m)-1);
    arma::vec ym = y.subvec(record, record+sizeM(m)-1);
    arma::vec betamini = betarecord.col(m-1);
    arma::mat Finv = inv(hessrecord+Loghess(Xm,betamini));
    double dism = 2*eps;
    int iterm = 0;
    arma::vec betam = betamini;
    while((dism>eps)&(iterm<maxstep)){
      betam = betamini+Finv*(Loggrad(Xm,ym,betamini)+hessrecord*(betarecord.col(m-1)-betamini));
      dism = sum(abs(betam-betamini));
      betamini = betam;
      iterm = iterm+1;
    }
    betarecord.col(m) = betam;
    hessrecord = hessrecord+Loghess(Xm, betam);
    record = record + sizeM(m);
  }
  return(betarecord.col(M-1));
}
// [[Rcpp::export]]
Rcpp::List SQRcppW(const arma::mat& Z, arma::vec Y, const double tau = 0.5, double h = 0.05, const double constTau = 1.345, 
                   const double tol = 0.0001, const int iteMax = 5000, const double stepMax = 100.0) {
  const int n = Z.n_rows, p = Z.n_cols;
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::vec gradOld(p), gradNew(p), der(n);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax, stepMax);
  arma::vec res = Y - Z * beta;
  updateGauss(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateGauss(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), stepMax);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateGauss(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List seqSQRcppW(const arma::mat& Z, arma::vec Y, arma::vec sizeM, const double tau = 0.5, double h = 0.05, const double constTau = 1.345, 
                      const double tol = 0.0001, const int iteMax = 5000, const double stepMax = 100.0, double inisize = 2000) {
  const int n = Z.n_rows, p = Z.n_cols, M = sizeM.size();
  const double h1 = 1.0 / h;
  double nm1 = 1.0 / inisize, record = sizeM(0);
  arma::mat betaiter(p, M), hessiter(p, p), hess(p,p);
  arma::vec gradOld(p), gradNew(p), der1(inisize), iterecord(M-1);
  arma::mat Z1 = Z.rows(0, inisize-1);
  arma::vec Y1 = Y.subvec(0, inisize-1);
  arma::vec beta = SQRcppW(Z1, Y1, tau, h)["coeff"];
  betaiter.col(0) = beta;
  arma::vec res1 = Y1 - Z1 * beta;
  hessGauss(Z1, res1, der1, hess, tau, nm1, h1);
  hessiter = hess;
  for(int m = 1; m < M; m++){
    
    double hm = 1/std::max(std::pow((std::log(n*(m+1)/M))*(n*(m+1)/M), 0.25), 0.05);
    double hm1 = 1/hm;
    arma::mat Zm = Z.rows(record, record+sizeM(m)-1);
    arma::vec Ym = Y.subvec(record, record+sizeM(m)-1);
    arma::vec resm = Ym - Zm * beta;
    arma::vec derm(sizeM(m));
    nm1 = 1.0 / sizeM(m);
    updateGauss(Zm, resm, derm, gradOld, tau, nm1, hm1);
    beta -= gradOld;
    arma::vec betaDiff = -gradOld;
    resm -= Zm * betaDiff;
    updateGauss(Zm, resm, derm, gradNew, tau, nm1, hm1);
    gradNew += hessiter*(beta - betaiter.col(m-1));
    arma::vec gradDiff = gradNew - gradOld;
    int ite = 1;
    while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
      double alpha = 1.0;
      double cross = arma::as_scalar(betaDiff.t() * gradDiff);
      if (cross > 0) {
        double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
        double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
        alpha = std::min(std::min(a1, a2), stepMax);
      }
      gradOld = gradNew;
      betaDiff = -alpha * gradNew;
      beta += betaDiff;
      resm -= Zm * betaDiff;
      updateGauss(Zm, resm, derm, gradNew, tau, nm1, hm1);
      gradNew += hessiter*(beta- betaiter.col(m-1));
      gradDiff = gradNew - gradOld;
      ite++;
    }
    betaiter.col(m) = beta;
    hessGauss(Zm, resm, derm, hess, tau, nm1, hm1);
    hessiter += hess;
    record += sizeM(m);
    iterecord(m-1) = ite;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("record") = betaiter, Rcpp::Named("ite") = iterecord, Rcpp::Named("bandwidth") = h);
}

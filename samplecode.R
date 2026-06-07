n = 50000
p = 50
set.seed(999)
X = matrix(rnorm(n*p), n, p) 
X[,(0.8*p+1):p] = matrix(rbinom(0.2*p*n, 1, 0.5), n, 0.2*p)
Xinter = cbind(1, X)
beta = c(-3, rep(c(1, -1), 0.4*p), rep(1, 0.2*p))
d = 5
beta_mp = c(1, 0.01, rep(0.4, d), rep(0, p-d))
p_mp = length(beta_mp)
MPV = rep(0, nsim)
K = 50
epsADM = 5e-02
epsN = 1e-02
maxstep = 5000
maxstepN = 100
sigma = 5

e = rnorm(n)
y = Xinter%*%beta+(sigma+X[,1])*e
Z = cbind(y, X)
Zinter = cbind(1, Z)
mp = exp(Zinter%*%beta_mp)/(1+exp(Zinter%*%beta_mp))
R = rbinom(n, 1, mp)
MPV[i] = 1-sum(R)/n
yfull = y[R==1]
Xfull = X[R==1,]
Xinterfull = Xinter[R==1,]


##IPW-ADMM
result_mp_paraADMMW = paraBetampCPP(Zinter, R, K, epsN, maxstepN)
beta_mp_paraADMMW = result_mp_paraADMMW[1:p_mp]
time_paraADMMW_pre = result_mp_paraADMMW[p_mp+1]
mp_paraADMMW = exp(Zinter%*%beta_mp_paraADMMW)/(1+exp(Zinter%*%beta_mp_paraADMMW))
mp_paraADMMW_full = mp_paraADMMW[R==1]
yfull_paraADMMW = yfull/mp_paraADMMW_full
Xfull_paraADMMW = Xfull/mp_paraADMMW_full
Xinterfull_paraADMMW = Xinterfull/mp_paraADMMW_full

nk = n/K
numk = rep(0, K)
for(m in 1:K){
  numk[m] = sum(R[(1+nk*(m-1)):(nk*m)])
}

tau = 0.3
beta_true = c(beta[1]+sigma*qnorm(tau), beta[2]+qnorm(tau), beta[3:(p+1)])
result_paraADMMW3 = paraQRCPP(Xinterfull_paraADMMW, yfull_paraADMMW, numk, K, tau, rho = 1, epsADM, maxstep, FALSE)
beta_paraADMMW3 = result_paraADMMW3[1:(p+1)]
AE_paraADMMW3 = sum(abs(beta_paraADMMW3-beta_true))
time_paraADMMW3_com = result_paraADMMW3[p+2]
time_paraADMMW3 = time_paraADMMW_pre+time_paraADMMW3_com


###IPW-renewable
sizeK = rep(nk, K)
ptm = proc.time() 
result_mp_seqW = seqNewtoncpp(Zinter, R, rep(0,p_mp), sizeK)
time_seqW_pre = (proc.time() - ptm)[3]
beta_mp_seqW =  result_mp_seqW[1:p_mp]
mp_seqW = exp(Zinter%*%beta_mp_seqW)/(1+exp(Zinter%*%beta_mp_seqW))
mp_seqW_full = mp_seqW[R==1]
yfull_seqW = yfull/mp_seqW_full
Xfull_seqW = Xfull/mp_seqW_full
Xinterfull_seqW = Xinterfull/mp_seqW_full

ptm = proc.time() 
result_seqW3 = seqSQRcppW(Xinterfull_seqW, yfull_seqW, numk, tau = tau) 
time_seqW3_com = (proc.time() - ptm)[3]
beta_seqW3 = result_seqW3$coeff
AE_seqW3 = sum(abs(beta_seqW3-beta_true))
time_seqW3 = time_seqW_pre+time_seqW3_com

import os
os.chdir("/Users/christophersaw/Desktop/Replication")
import pandas as pd
import numpy as np
import linearmodels
from linearmodels import OLS
import statsmodels.api as sm
import scipy as sp
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df=pd.read_csv('df.csv')
n_obs=int(len(df))
n_firms=int(max(df['firm_id']))
n_periods=int(max(df['time']))

### Step 1: Labor-augmenting productivity

# Create variables for m-l and pm-w
df['ml']=df['m']-df['l']		# m_it - l_it
df['pmw']=df['pm']-df['w']		# pm_it - w_it

# OLS regression for equation (4)
dep_var=df['ml']
lin_var=sm.add_constant(df[['pmw']])
ols=OLS(dep_var, lin_var).fit()

# Recover sigma
s=-ols.params[1]

# Recover labor-augmenting productivity
df['WL_hat']=ols.resids/(1-s)
bl=np.exp(df.WL_hat)



### Step 2: Hicks-neutral productivity

# Non-parameteric regression (polynomial approximation)
# set degree of polynomial
poly=PolynomialFeatures(degree=3) 
dep_var=df['q']
# endogenous variables for polynominal regression:
poly_var=df[['k','l','m','pm1','p1','k1','m1','D1','z1','WL_hat']] 
poly_var=pd.DataFrame(poly.fit_transform(poly_var)) 
poly_reg=OLS(dep_var,poly_var).fit() # Regression of q on phi(k,l,m)
phi=np.array(poly_reg.predict()) # Calculate fitted values 

# Construct arrays
K=np.exp(df.k)
L=np.exp(df.l)
M=np.exp(df.m)

# Construct function f(x; theta)
def fx(theta):
	fx_theta=np.zeros((n_obs,1))
	X=np.zeros((n_obs,1))
	nu=theta[0]
	bk=theta[1]
	bm=theta[2]
	C=(nu*s)/(s-1)
	p=(s-1)/s
	for i in range(n_obs):
			X[i]=bk*np.power(K[i],p)+\
			np.power(bl[i]*L[i],p)+\
			bm*np.power(M[i],p)
			fx_theta[i]=C*np.log(X[i])
	return fx_theta


# Construct functions omega and xi
def omega(theta):
	omega_theta=phi-fx(theta) # this gives omega^H + xi^H
	return omega_theta


Z0=np.array(df.z)[0:n_obs-n_firms]
Z0=np.reshape(Z0,(n_obs-n_firms,1))
def xi(theta):
	W1=np.array(omega(theta))[n_firms:] # remove first n_firms in year 1
	W1=np.reshape(W1,(n_obs-n_firms,1))
	W0=np.array(omega(theta))[0:n_obs-n_firms] # remove last n_firms in year T
	W0=np.reshape(W0,(n_obs-n_firms,1))
	WZ=np.concatenate((W0,Z0),axis=1)
	beta=np.linalg.inv(WZ.transpose() @ WZ) @ WZ.transpose() @ W1 # OLS
	xi_theta=W1 - (WZ @ beta) # Calculate residuals xi^H
	return xi_theta


# Instruments for GMM
Z=np.array(sm.add_constant(df[['l1','m1','w1','pm1','D1']]))
Z=Z[n_firms:,0:] 

# GMM objective function
weightmatrix=np.linalg.inv(Z.transpose() @ Z) # optimal weighting matrix for GMM
def gmmobjfn(theta):
        gmmobjfn=xi(theta).transpose() @ Z @ weightmatrix @ Z.transpose() @ xi(theta)
        return gmmobjfn

# Solve for theta^H
theta0=np.array([[1],[0.5],[0.5]])
results=minimize(gmmobjfn,theta0,\
	method='Nelder-Mead',options={'maxiter':1000})

theta_H=results.x

# Compare estimates with simulated values
df=df.loc[df['time']!=1]	# Drop t=1
df['WH_hat']=omega(theta_H)[n_firms:]-xi(theta_H)

simH=df.WH                                                                                                                                                                                             
estH=df.WH_hat
OLS(simH,estH).fit()

simL=df.WL                                                                                                                                                                                             
estL=df.WL_hat
OLS(simL,estL).fit()


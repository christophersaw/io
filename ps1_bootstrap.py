import os
os.chdir("/Users/christophersaw/Desktop/PS1")
import pyreadr
import pandas as pd
import numpy as np
from numpy.random import randint
import linearmodels
from linearmodels import OLS
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from pystout import pystout
import scipy as sp
from sklearn.preprocessing import PolynomialFeatures
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set seed
np.random.seed(123456789)

# Load data
robj=pyreadr.read_r('acf_sim_V2.RData')
print(robj.keys()) # retrieve object name
df=robj['acf_sim'] # convert to dataframe

# (3g) Construct K bootstrap draws of the dataset

# NOTES:
# Want to construct K subsamples, each with Nk observations 'Nk x K'
# Since the dataset is a panel, we should take draws of firm_id with replacement
# Total number of draws = Nk x K / n_periods 
# Place all firm x time draws into 'dfb', dfb contains all 'Nk x K' bootstrap obs
# Then divide 'dfb' into 40 subsamples
# Each subsample consists of 50 firms, each observed for 20 periods

# Parameters for bootstrap
n_obs=int(len(df)) # 20,000
n_firms=int(max(df['firm_id'])) # 1000
n_periods=int(max(df['year'])) # 20
K=40
Nk=1000
n_draws=int(Nk*K/n_periods) # 2,000
nb_firms=int(Nk/n_periods) # no. of firms in each subsample

# Draw firms with replacement
draws=pd.DataFrame(randint(1,1+n_firms,(n_draws,1)))\
.rename(columns={0:'firm_id'}) # Draw 2,000 firms
dfb=pd.merge(draws,df,on='firm_id',how ='inner',validate="m:m") 

# Divide into subsamples and create an id for each subsample
n_subsample=pd.Series(np.linspace(1,K,num=K)) # create 40 subsamples
newcol=pd.DataFrame(n_subsample.repeat(Nk))\
.rename(columns={0:'subsample'}).reset_index() # each with Nk obs

# Join subsample ids onto dfb
dfb=pd.merge(dfb,newcol['subsample'],left_index=True, right_index=True)

# (3h) Recover K estimates of beta

# Functions for GMM (dimensions adjusted for subsample)
def f(beta):
	f_beta=np.zeros((Nk,1))
	for i in range(Nk):
			f_beta[i]=np.sum(X[i,j]*beta[j] for j in range(0,3))
	return f_beta

def omega(beta):
	omega_beta=phi_it-f(beta) # use results from (3a),(3b)
	return omega_beta

def xi(beta):
	W1=np.array(omega(beta))[nb_firms:]
	W0=np.array(omega(beta))[0:Nk-nb_firms] 
	rho=np.linalg.inv(W0.transpose() @ W0) @ W0.transpose() @ W1 
	xi_beta=W1-rho*W0 
	return xi_beta

def gmmobjfn(beta):
	gmmobjfn=xi(beta).transpose() @ Z @ W @ Z.transpose() @ xi(beta)
	return gmmobjfn

# Use same initial guess for all subsamples
beta_init=np.array([[0.4083],[0.0766],[0.8952]])

# Create a new firm_id that is unique within each subsample
new_firm_id=pd.Series(np.linspace(1,nb_firms,num=nb_firms))
newcol2=pd.DataFrame(new_firm_id.repeat(n_periods))\
.rename(columns={0:'new_firm_id'}).reset_index()

# Create a box to store our bootstrap estimates of beta
bootstrap_box=np.zeros((K,3))

# Bootstrap estimates of beta, loop over k
# Within each loop, construct X, Z, W 
# and estimate phi, f, omega, xi
for k in range(K):
	dfk=dfb.loc[dfb['subsample']==k+1].reset_index() # keep only observations in k
	dfk=pd.merge(dfk,newcol2['new_firm_id'],\
	left_index=True, right_index=True) # use new_firm_id
	dfk=dfk.sort_values(by=['year','new_firm_id']) # sort data to set panel
	dep_var=dfk['q'] # dependent variable
	poly=PolynomialFeatures(degree=3) # set degree of polynomial
	poly_var=dfk[['k','labor','m']] # endogenous variables for polynominal regression
	poly_var=pd.DataFrame(poly.fit_transform(poly_var))
	poly_reg=OLS(dep_var,poly_var).fit() 
	phi_it=np.array(poly_reg.predict())
	X=np.array(sm.add_constant(dfk[['k','labor']])) # endogenous variables
	Z=np.array(sm.add_constant(dfk[['k','labor_1']])) #instruments
	Z=Z[nb_firms:,0:] # remove first nb_firm observations in year 1
	W=np.linalg.inv(Z.transpose() @ Z) # optimal weighting matrix for GMM
	results=sp.optimize.minimize(gmmobjfn,beta_init,\
		method='Nelder-Mead',options={'maxiter':1000})
	beta_acf=results.x
	bootstrap_box[k,:]=beta_acf


# (3h) Use bootstrap estimates to calculate 90% CI
c=1.645
bootstraps=pd.DataFrame(bootstrap_box).rename(columns={0:'beta_0',1:'beta_k',2:'beta_l'})
bk_bar=bootstraps.beta_k.mean() # mean of bootstrap estimates of beta_k
bk_s=bootstraps.beta_k.std() # std dev of bootstrap estimates
beta_k_LB=bk_bar-(c*bk_s/np.sqrt(K)) # LB for 90% CI
beta_k_UB=bk_bar+(c*bk_s/np.sqrt(K)) # UB for 90% CI
bl_bar=bootstraps.beta_l.mean() # repeat above for beta_l
bl_s=bootstraps.beta_l.std()
beta_l_LB=bl_bar-(c*bl_s/np.sqrt(K))
beta_l_UB=bl_bar+(c*bl_s/np.sqrt(K))

# Export estimates to csv
bootstraps.to_csv('bootstraps.csv',index=False)
pd.DataFrame([beta_k_LB,beta_k_UB])\
.rename({0:'CI beta_k'},axis=1).rename({0:'LB',1:'UB'},axis=0)\
.to_csv('beta_k_CI.csv',index=False)
pd.DataFrame([beta_l_LB,beta_l_UB]).rename({0:'CI beta_l'},axis=1)\
.rename({0:'LB',1:'UB'},axis=0)\
.to_csv('beta_l_CI.csv',index=False)

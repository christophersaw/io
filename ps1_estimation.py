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
n_obs=int(len(df))
n_firms=int(max(df['firm_id']))
n_periods=int(max(df['year']))
panel_df=df.set_index(['firm_id', 'year']) # set entity and time indices

# (1) OLS and (2) FE regressions
dep_var=panel_df['q']
lin_var=sm.add_constant(panel_df[['k','labor']])
ols=OLS(dep_var, lin_var).fit()
fe=PanelOLS(dep_var, lin_var, entity_effects=True)\
.fit(cov_type="clustered", cluster_entity=True)
# Save results
pystout(models=[ols,fe],
        file='ols_fe.tex',
        digits=3,
        mgroups={'OLS':1,'FE':2},
        varlabels={'const':'Constant','k':'Capital','labor':'Labor'},
        modstat={'nobs':'Obs','rsquared':'R\sym{2}'},
        addnotes=['Standard errors in parentheses',
        '\sym{**} p\sym{_<}0.01, * p\sym{_<}0.05, \sym{+} p\sym{_<}0.10']
        )

# (3a) Non-parameteric regression (polynomial approximation)
poly=PolynomialFeatures(degree=3) # set degree of polynomial
poly_var=panel_df[['k','labor','m']] # endogenous variables for polynominal regression
poly_var=pd.DataFrame(poly.fit_transform(poly_var)) # calculate terms up to third degree
poly_reg=OLS(dep_var,poly_var).fit() # Regression of q on phi(k,l,m)
phi_it=np.array(poly_reg.predict()) # Calculate fitted values 

# (3b) Construct function f(k,l; b) = b0 + bl*l + bk*k
X=np.array(sm.add_constant(panel_df[['k','labor']])) 
def f(beta):
	f_beta=np.zeros((n_obs,1))
	for i in range(n_obs):
			f_beta[i]=np.sum(X[i,j]*beta[j] for j in range(0,3))
	return f_beta

# (3c) Construct functions for omega and xi
def omega(beta):
	omega_beta=phi_it-f(beta) # use results from (3a),(3b)
	return omega_beta

def xi(beta):
	W1=np.array(omega(beta))[n_firms:] # remove first n_firm observations in year 1
	W0=np.array(omega(beta))[0:n_obs-n_firms] # remove last n_firm observations in year T
	rho=np.linalg.inv(W0.transpose() @ W0) @ W0.transpose() @ W1 # OLS
	xi_beta=W1-rho*W0 # Calculate residuals xi
	return xi_beta

# (3d) Assemble instruments Z=[1 k l_1]
Z=np.array(sm.add_constant(panel_df[['k','labor_1']]))
Z=Z[n_firms:,0:] # remove first n_firm observations in year 1

# (3e) GMM objective function
W=np.linalg.inv(Z.transpose() @ Z) # optimal weighting matrix for GMM
def gmmobjfn(beta):
        gmmobjfn=xi(beta).transpose() @ Z @ W @ Z.transpose() @ xi(beta)
        return gmmobjfn

# (3f) Solve for beta
beta_init=np.array([[0.4083],[0.0766],[0.8952]]) # use beta_FE
results=sp.optimize.minimize(gmmobjfn,beta_init,\
	method='Nelder-Mead',options={'maxiter':1000})
 # final_simplex: (array([[0.45077117, 0.03287014, 0.96821934],
 #       [0.45084014, 0.03284315, 0.96825184],
 #       [0.45086032, 0.03280441, 0.9682736 ],
 #       [0.45082943, 0.03283342, 0.96822781]]), 
 #		 array([6.48237678e-06, 6.86708258e-06, 7.77195868e-06, 8.16875677e-06]))
 #           fun: 6.4823767804439994e-06
 #       message: 'Optimization terminated successfully.'
 #          nfev: 109
 #           nit: 61
 #        status: 0
 #       success: True
 #             x: array([0.45077117, 0.03287014, 0.96821934])
beta_acf=results.x
gmm_value=results.fun

# (5) Tabulate and export results
beta_ols=pd.DataFrame(ols.params).reset_index()
beta_fe=pd.DataFrame(fe.params).reset_index()
beta_acf=pd.DataFrame(beta_acf.transpose())
coeffs=beta_ols.merge(beta_fe,on='index').join(beta_acf)\
.rename(columns={'index':'','parameter_x':'OLS','parameter_y':'FE',0:'ACF'})
coeffs.to_csv('coeffs.csv',index=False)
# coeffs
#                OLS        FE       ACF
# 0  const  0.391694  0.408290  0.450771
# 1      k  0.079261  0.076602  0.032870
# 2  labor  0.930155  0.895151  0.968219

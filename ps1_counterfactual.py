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
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load estimates
coeffs=pd.read_csv('coeffs.csv')
beta=np.array(coeffs)[:,3]

# Repeat steps (3a) to (3c) with ACF estimates
dep_var=panel_df['q']

# (3a) Non-parameteric regression (polynomial approximation)
poly=PolynomialFeatures(degree=3) # set degree of polynomial
poly_var=panel_df[['k','labor','m']] # endogenous variables for polynominal regression
poly_var=pd.DataFrame(poly.fit_transform(poly_var)) # calculate terms up to third degree
poly_reg=OLS(dep_var,poly_var).fit() # Regression of q on phi(k,l,m)
phi_it=np.array(poly_reg.predict()) # Calculate fitted values 

# (3b) Construct function f(k,l; b) = b0 + bl*l + bk*k
X=np.array(sm.add_constant(panel_df[['k','labor']])) 
f_beta=np.zeros((n_obs,1))
for i in range(n_obs):
	f_beta[i]=np.sum(X[i,j]*beta[j] for j in range(0,3))


# (3c) Calculate productiivty omega and merge onto dataframe
omega=pd.DataFrame(phi_it-f_beta).rename(columns={0:'omega'})
df=pd.merge(df,omega,left_index=True, right_index=True)

# Keep only observations at T
T=int(max(df['year']))
dft=df.loc[df['year']==T].reset_index()

# (1) Plot empirical distribution of productivity at T
dft['W']=np.exp(dft['omega'])
sns.distplot(dft['W'], hist=False, kde=True, bins=int(20), color='darkblue')
plt.xlabel('Productivity Levels')
plt.ylabel('Measure of Firms')
plt.savefig('productivity(levels).pdf')
plt.close('all')

# Convert q, k, l from logs to levels
dft['Q']=np.exp(dft['q'])
dft['K']=np.exp(dft['k'])
dft['L']=np.exp(dft['labor'])

# Calculate A = exp(b0)*exp(w_it)*exp(e_it)
dft['epsilon']=dft['q']-beta[0]-beta[1]*dft['k']-beta[2]*dft['labor']-dft['omega']
dft['A']=np.exp(beta[0])*np.exp(dft['omega'])*np.exp(dft['epsilon'])

# (2a) Plot marginal product of capital at T
dft['MPK']=dft['A']*beta[1]*np.power(dft['K'],(beta[1]-1))*np.power(dft['L'],beta[2])
sns.distplot(dft['MPK'], hist=False, kde=True, bins=int(20), color='red')
plt.xlabel('Marginal Product of Capital')
plt.ylabel('Measure of Firms')
plt.savefig('MPK.pdf')
plt.close('all')

# (2b) Average MPK across firms at T
avg_mpk=dft['MPK'].mean()

# (2c) First and ninth deciles of MPK distribution
dft['MPK_decile']=pd.qcut(dft['MPK'], 10, labels=False)
decile_1_mpk=dft.loc[dft['MPK_decile']==0,'MPK'].mean()
decile_9_mpk=dft.loc[dft['MPK_decile']==8,'MPK'].mean()
# Save results
mpk_stats=pd.DataFrame(np.array([[avg_mpk],[decile_1_mpk],[decile_9_mpk]]))\
.rename({0:'MPK'},axis=1).rename({0:'Average',1:'First Decile',2:'Ninth Decile'},axis=0)
mpk_stats.to_csv('MPK.csv')
#                    MPK
# Average       0.019624
# First Decile  0.005749
# Ninth Decile  0.031467

# (3a) Total economy capital stock and output
total_Q=dft['Q'].sum()
total_K=dft['K'].sum()
print(total_K)

# (3b) Capital reallocation
# Create arrays and parameters for capital reallocation
MPK=np.array(dft['MPK'])
K=np.array(dft['K'])
L=np.array(dft['L'])
A=np.array(dft['A'])
new_MPK=np.zeros((n_firms,1))
max_MPK=np.max(MPK)
step=1
# Suppose initial capital allocation for every firm is set to K min
new_K=np.ones((n_firms,1))*np.min(K)
new_total_K=np.sum(new_K)
# Iterate
while new_total_K < total_K:
	i=np.where(MPK==max_MPK)
	new_K[i]=K[i]+step
	new_MPK[i]=A[i]*beta[1]*np.power(new_K[i],(beta[1]-1))*np.power(L[i],beta[2])
	K[i]=new_K[i]
	MPK[i]=new_MPK[i]
	max_MPK=np.max(MPK)
	new_total_K=np.sum(new_K)
	new_total_K


# (3c) Marginal product of capital
for i in range(n_firms):
	new_MPK[i]=A[i]*beta[1]*np.power(new_K[i],(beta[1]-1))*np.power(L[i],beta[2])


MPK_stats=np.array([np.min(new_MPK),np.mean(new_MPK),np.max(new_MPK)])
MPK_stats=pd.DataFrame(MPK_stats)\
.rename({0:'MPK'},axis=1)\
.rename({0:'min',1:'mean',2:'max'},axis=0)
MPK_stats.to_csv('Counterfactual_MPK.csv',index=False)
#            MPK
# min   0.003446
# mean  0.041816
# max   0.679497

# Calculate counterfactual output
dft['new_K']=new_K # Join new capital allocation to dataframe
dft['new_Q']=dft['A']*np.power(dft['new_K'],beta[1])*np.power(dft['L'],beta[2])
new_total_Q=dft['new_Q'].sum() 
output_compare=np.array([total_Q,new_total_Q])
output_compare=pd.DataFrame(output_compare)\
.rename({0:'Total Output'},axis=1)\
.rename({0:'Data',1:'Counterfactual'})
output_compare.to_csv('Counteractual_Output.csv',index=False)
#                 Total Output
# Data             3652.512807
# Counterfactual   3648.744031

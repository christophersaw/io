import os
os.chdir("/Users/christophersaw/Desktop/PS2")
import pyreadr
import pandas as pd
import numpy as np
import linearmodels
from linearmodels import OLS
from pystout import pystout
import statsmodels as sm
from statsmodels.discrete.discrete_model import Probit
import scipy as sp
from sklearn.preprocessing import PolynomialFeatures
import random
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# (1) Static Cournot

# Load data
robj=pyreadr.read_r('cournot.RData')
df=robj['cournot'] # convert to dataframe

# (1a) Summary statistics
p_stats=df['p'].describe() # Market price
p_stats=pd.DataFrame(p_stats)
Q_stats=df['Q'].describe() # Market quantity
Q_stats=pd.DataFrame(Q_stats)
firm_q_stats=pd.DataFrame(df['q_1']).join(df['q_2']).join(df['q_3']).melt().describe() # Firm quantity
firm_q_stats=pd.DataFrame(firm_q_stats).rename(columns={'value':'q'})
firm_omega_stats=pd.DataFrame(df['omega_1']).join(df['omega_2']).join(df['omega_3']).melt().describe() # Firm productivity
firm_omega_stats=pd.DataFrame(firm_omega_stats).rename(columns={'value':'omega'})
stats=p_stats.join(Q_stats)
stats=stats.join(firm_q_stats)
stats=stats.join(firm_omega_stats)
# stats.to_csv('sumstats.csv',index=True) # save results

# (1b) Estimate elasticity of demand
df['log_p']=np.log(df.p)
df['log_Q']=np.log(df.Q)
ols=OLS(df['log_p'], df['log_Q']).fit()
beta=float(ols.params)
theta_hat=-1/beta #2.006987
# pystout(models=[ols],
#         file='ols.tex',
#         digits=3,
#         modstat={'nobs':'Obs','rsquared':'R\sym{2}'},
#         addnotes=['Standard errors in parentheses',
#         '\sym{**} p\sym{_<}0.01, * p\sym{_<}0.05, \sym{+} p\sym{_<}0.10']
#         )

# (1c) Calculate alpha
# Keep observations where omegas are perfectly symmetric
s1=np.array(df['omega_1'])
s2=np.array(df['omega_2'])
s3=np.array(df['omega_3'])
n=len(s1)
ind=np.zeros((n,))
for i in range (n):
	if s1[i]==s2[i]==s3[i]:
		ind[i]=1
	else:
		ind[i]=0

df['ind']=ind
df2=df.loc[df['ind']==1]

# Reshape dataframe from wide to long
mkt_p=pd.DataFrame([df2.p,df2.p,df2.p]).transpose().melt().rename(columns={'value':'p'})
mkt_p=pd.DataFrame(mkt_p.p)
mkt_Q=pd.DataFrame([df2.Q,df2.Q,df2.Q]).transpose().melt().rename(columns={'value':'Q'})
mkt_Q=pd.DataFrame(mkt_Q.Q)
firm_q=pd.DataFrame(df2['q_1']).join(df2['q_2']).join(df2['q_3']).melt().rename(columns={'value':'q'})
firm_q=pd.DataFrame(firm_q.q)
firm_omega=pd.DataFrame(df2['omega_1']).join(df2['omega_2']).join(df2['omega_3']).melt().rename(columns={'value':'omega'})
firm_omega=pd.DataFrame(firm_omega.omega)
df_long=mkt_p.join(mkt_Q).join(firm_q).join(firm_omega)

# Calculate LHS of equation 1
df_long['LHS'] = (1 + beta*(df_long.q/df_long.Q))*df_long.p

# Calculate alpha_hat
df_long['alpha'] = (df_long.LHS*df_long.omega)/(1-df_long.LHS)
alpha_hat=np.mean(df_long.alpha)

# Calculate marginal cost
df['c_1']=alpha_hat/(alpha_hat+df['omega_1'])
df['c_2']=alpha_hat/(alpha_hat+df['omega_2'])
df['c_3']=alpha_hat/(alpha_hat+df['omega_3'])

# Calculate Q_1 based on omega_1 (q_1 and omega_1 do not correspond)
df['Q_1']=df.Q*theta_hat*(1-(alpha_hat/(alpha_hat+df['omega_1']))*(1/df.p))

# Calculate profit for i = 1
df['profit_1']=(df['p']-df['c_1'])*df['Q_1']
df.loc[df['omega_1']==0,'c_1']=0 # c_1=0 if omega_1 = 0 
df.loc[df['omega_1']==0,'Q_1']=0 # Q_1=0 if omega_1 = 0 
df.loc[df['omega_1']==0,'profit_1']=0 # profit_1=0 if omega_1 = 0 

# Profit as a function of the state for firm i = 1
df['state1']=df.omega_1.astype(str)+df.omega_2.astype(str)+df.omega_3.astype(str)
states1=df['state1'].to_numpy()
df['state2']=df.omega_1.astype(str)+df.omega_3.astype(str)+df.omega_2.astype(str) #rivals can be in any order
states2=df['state2'].to_numpy()
profits=df['profit_1'].to_numpy()


# (1d) Load states
robj2=pyreadr.read_r('states.RData')
df_states=robj2['states'] # convert to dataframe
df_states=df_states.rename(columns={0:'s1',1:'s2',2:'s3'})

# (1e) Tabulate profits over states

# Profit function
def pi(s1,s2,s3):
        if s1==0 & s2==0 & s3==0:
                value=0
        else:
                s=str(s1)+str(s2)+str(s3)
                i=np.array(np.where(states1==str(s)))
                j=np.array(np.where(states2==str(s)))
                if i > 0:
                        value=float(profits[i])
                else:
                        value=float(profits[j])
        return value

n=len(df_states)
S=np.array(df_states)
pi1=np.zeros((n,1))
pi2=np.zeros((n,1))
pi3=np.zeros((n,1))
for i in range(n):
	pi1[i]=pi(S[i,0],S[i,1],S[i,2])
	pi2[i]=pi(S[i,1],S[i,0],S[i,2])
	pi3[i]=pi(S[i,2],S[i,1],S[i,0])

df_states['pi1']=pi1
df_states['pi2']=pi2
df_states['pi3']=pi3

df_cournot=df

# (2) Policy Functions

# (2a) Load data
robj=pyreadr.read_r('bbl.RData')
data=robj['bbl'] # convert to dataframe


# (2b) Probability of entry
df=data
df1=data.loc[df['omega_1']==0].reset_index() # New dataframe where all s1 = 0
# Create 2nd degree polynomial
poly=PolynomialFeatures(degree=2) # set degree of polynomial
poly_var=df1[['omega_2','omega_3']] # endogenous variables for polynominal regression
poly_var=pd.DataFrame(poly.fit_transform(poly_var)) # calculate terms up to second degree
poly_var=poly_var.rename(columns={0:'Constant',1:'s2',2:'s3',3:'s2^2',4:'s2s3',5:'s3^2'})
# Probit regression to calculate p_enter
dep_var=df1['entry']
p_reg_enter=Probit(dep_var,poly_var).fit() # Regression of entry on poly(s2,s3)

def p_enter(s1,s2,s3):
        p_enter1=p_reg_enter.predict([1,s2,s3,np.square(s2),s2*s3,np.square(s3)])
        return p_enter1


# (2c) Probability of exit
df2=data
df2['omega_sum']=df2.omega_1+df2.omega_2+df2.omega_3
df2=df2.loc[df2['omega_sum']!=0].reset_index() # Remove cases where no firm is in the market
# Create 2nd degree polynomial
poly=PolynomialFeatures(degree=2) # set degree of polynomial
poly_var2=df2[['omega_1','omega_2','omega_3']] # endogenous variables for polynominal regression
poly_var2=pd.DataFrame(poly.fit_transform(poly_var2)) # calculate terms up to second degree
poly_var2=poly_var2.rename(columns={0:'Constant',1:'s1',2:'s2',3:'s3',4:'s1^2',5:'s1s2',6:'s1s3',7:'s2^2',8:'s2s3',9:'s3^2'})
# Probit regression to calculate p_exit
dep_var=df2['exit']
p_reg_exit=Probit(dep_var,poly_var2).fit() # Regression of exit on poly(s1,s2,s3)

# Policy function for exit
def p_exit(s1,s2,s3):
        p_exit1=p_reg_exit.predict([1,s1,s2,s3,np.square(s1),s1*s2,s1*s3,np.square(s2),s2*s3,np.square(s3)])
        return p_exit1


# (3) Forward simulation
K=2
T=50
NS=np.shape(S)[0]
NF=np.shape(S)[1]
ST=np.zeros((NS,NF,T+1,K))
X=np.zeros((NS,NF,T+1,K))       # optimal exit decision

# (3a) Simulate K paths over T periods, each path starting at s0; record any exits
for k in range(K):
        ST[:,:,0,k]=S
        for t in range(T):
                for i in range(NS):
                        for j in range(NF):
                                x1=ST[i,j,t,k]
                                if j==0:
                                        x2=ST[i,j+1,t,k]
                                        x3=ST[i,j+2,t,k]
                                elif j==1:
                                        x2=ST[i,j-1,t,k]
                                        x3=ST[i,j+1,t,k]
                                elif j==2:
                                        x2=ST[i,j-2,t,k]
                                        x3=ST[i,j-1,t,k]
                                if ST[i,j,t,k]==0:        
                                        if np.random.uniform(0,1) < p_enter(x1,x2,x3):
                                                ST[i,j,t+1,k]=random.choice(range(1,7))
                                        else:
                                                ST[i,j,t+1,k]=ST[i,j,t,k]
                                else:
                                        if ST[i,j,t,k]==np.min(ST[i,:,t,k]):
                                                if np.random.uniform(0,1) < p_exit(x1,x2,x3):
                                                        ST[i,j,t+1,k]=0
                                                        X[i,j,t,k]=1
                                                else:
                                                        ST[i,j,t+1,k]=ST[i,j,t,k]
                                        else:
                                                ST[i,j,t+1,k]=ST[i,j,t,k]


# (3b) Value Function for (3e) each initial state s; for firm i = 1
def EV(x):
        Exit=x
        beta=0.95
        vt=np.zeros((NS,T+1,K,2)) # per-period profit or exit choice 
        V=np.zeros((NS,K)) # present discounted value, for equation 3
        EV=np.zeros((NS,1)) # average over K iterations, for each initial state s
        for k in range(K):
                for i in range (NS):
                        for t in range(T):
                                x1=ST[i,0,t,k].astype(int)
                                x2=ST[i,1,t,k].astype(int)
                                x3=ST[i,2,t,k].astype(int)
                                vt[i,t,k,0]=np.power(beta,t)*(1-Exit[i,0,t,k])*pi(x1,x2,x3)         
                                vt[i,t,k,1]=np.power(beta,t)*Exit[i,0,t,k]                          
                        V[i,k]=np.sum(vt[i,:,k,0])+np.sum(vt[i,:,k,1]) 
        for i in range(NS):
                EV[i]=np.mean(V[i,:])
        return EV

# (3c) Generate M perturbations of optimal exit and calculate EV for each m
M=2
bracket=[-5,-4,-3,-2,-1,1,2,3,4,5]
XM=np.zeros((NS,NF,T+1,K,M))
for m in range(M):
        for k in range(K):
                for t in range(5,T-4):
                        if X[i,0,t,k]==1:
                                tau=t+random.choice(bracket)
                                XM[i,0,t,k,m]=0
                                XM[i,0,tau,k,m]=1

EVM=np.zeros((NS,M))
for m in range(M):
        x=XM[:,:,:,:,m]
        EVM[:,m]=EV(x)


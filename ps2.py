caffeinate -is python3

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
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.stats import norm
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
K=200
T=50
NS=np.shape(S)[0]
NF=np.shape(S)[1]
ST=np.zeros((NS,NF,T+1,K))
X=np.zeros((NS,NF,T+1,K))

# (3a) Simulate K paths over T periods, each path starting at s0; record any exits in matrix X
for k in range(K):
        ST[:,:,0,k]=S
        for t in range(T):
                for i in range(NS):
                        for j in range(NF):
                                x1=ST[i,j,t,k] # p_enter and p_exit depends on the ordering of (s1,s2,s3)
                                if j==0:
                                        x2=ST[i,j+1,t,k]
                                        x3=ST[i,j+2,t,k]
                                elif j==1:
                                        x2=ST[i,j-1,t,k]
                                        x3=ST[i,j+1,t,k]
                                elif j==2:
                                        x2=ST[i,j-2,t,k]
                                        x3=ST[i,j-1,t,k]
                                if x1==0: # if firm j in state i is a potential entrant
                                        if np.random.uniform(0,1) < p_enter(x1,x2,x3): # chooses whether to enter
                                                ST[i,j,t+1,k]=random.choice(range(1,7)) # productivity is a random uniform draw
                                        else:
                                                ST[i,j,t+1,k]=ST[i,j,t,k]
                                else: # firm j in state i is an incumbent
                                        if (x2>0) & (x3>0): # and there are 3 incumbents (elifs consider other cases)
                                                if x1==np.min((x1,x2,x3)): # lowest productivity firm 
                                                        if np.random.uniform(0,1) < p_exit(x1,x2,x3): # chooses whether to exit
                                                                ST[i,j,t+1,k]=0
                                                                X[i,j,t,k]=1
                                                        else:
                                                                ST[i,j,t+1,k]=ST[i,j,t,k]
                                                else:
                                                        ST[i,j,t+1,k]=ST[i,j,t,k]
                                        elif (x2>0) & (x3==0):
                                                if x1<x2: # lowest productivity firm 
                                                        if np.random.uniform(0,1) < p_exit(x1,x2,x3): 
                                                                ST[i,j,t+1,k]=0 
                                                                X[i,j,t,k]=1
                                                        else:
                                                                ST[i,j,t+1,k]=ST[i,j,t,k]
                                                else:
                                                        ST[i,j,t+1,k]=ST[i,j,t,k]
                                        elif (x3>0) & (x2==0):
                                                if x1<x3: # lowest productivity firm 
                                                        if np.random.uniform(0,1) < p_exit(x1,x2,x3):
                                                                ST[i,j,t+1,k]=0
                                                                X[i,j,t,k]=1
                                                        else:
                                                                ST[i,j,t+1,k]=ST[i,j,t,k]
                                                else:
                                                        ST[i,j,t+1,k]=ST[i,j,t,k]
                                        elif (x3==0) & (x2==0):
                                                if np.random.uniform(0,1) < p_exit(x1,x2,x3): 
                                                        ST[i,j,t+1,k]=0 
                                                        X[i,j,t,k]=1
                                                else:
                                                        ST[i,j,t+1,k]=ST[i,j,t,k]


# Exit strategies
X1=np.zeros((NS,NF,T+1,K))  # optimal exit decision given initial state s
XT=np.zeros((NS,NF,T+1,K))  # indicator whether the firm has exited in any t
for k in range(K):
        for i in range(NS):
                for j in range(NF):
                        for t in range(T):
                                if np.sum(X1[i,j,:,k])==0:
                                        X1[i,j,t,k]=X[i,j,t,k]
                                        XT[i,j,t+1,k]=X[i,j,t,k] 
                                else:
                                        XT[i,j,t+1,k]=1

# Vector for equilibrum play (equation 3)
def V(exit_decision,exit_indicator):
        Exit=exit_decision #X1
        Ind=exit_indicator #XT
        beta=0.95
        v=np.zeros((NS,T+1,K,2)) # Vector in equation 3
        v_sum=np.zeros((NS,K,2)) # present discounted value, for equation 3
        v_avg=np.zeros((NS,2)) # average over K iterations, for each initial state s
        for k in range(K):
                for i in range (NS):
                        for t in range(T):
                                x1=ST[i,0,t,k].astype(int)
                                x2=ST[i,1,t,k].astype(int)
                                x3=ST[i,2,t,k].astype(int)
                                v[i,t,k,0]=np.power(beta,t)*(1-Ind[i,0,t,k])*pi(x1,x2,x3) 
                                v[i,t,k,1]=np.power(beta,t)*Exit[i,0,t,k]
                        v_sum[i,k,0]=np.sum(v[i,:,k,0])
                        v_sum[i,k,1]=np.sum(v[i,:,k,1]) 
        for i in range(NS):
                v_avg[i,0]=np.mean(v_sum[i,:,0])
                v_avg[i,1]=np.mean(v_sum[i,:,1])
        return v_avg

# (3c) Generate M perturbations 
M=200
X1m=np.zeros((NS,NF,T+1,K,M))
XTm=np.zeros((NS,NF,T+1,K,M))
for m in range(M):
        for k in range(K):
                for i in range(NS):
                        for t in range(5,T-5):
                                if X1[i,0,t,k]==1:
                                        tau=t+random.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
                                        X1m[i,0,t,k,m]=0
                                        X1m[i,0,tau,k,m]=1
                                        XTm[i,0,tau+1,k,m]=1
                        for t in range(T):
                                if XTm[i,0,t,k,m]==1:
                                        XTm[i,0,t+1,k,m]=1



# (3d) Calculate phi from moment inequalities

# Equilibrium play
eq_v=V(X1,XT)

# Off-equilibrium play
offeq_v=np.zeros((NS,2,M))
for m in range(M):
        x1=X1m[:,:,:,:,m]
        xt=XTm[:,:,:,:,m]
        offeq_v[:,:,m]=V(x1,xt)


# Objective function for moment inequality
def objfn(phi):
        w=v0-v1                         # v0 is eqn(3) under equilibrium play
        fn=w[0]+w[1]*phi
        if phi >= 0:
                if fn >= 0:             # if phi satisfies the moment inequality
                        return fn
                else:
                        return 9999     # if phi does not satisfy the moment inequality
        else:
                if fn <= 0:             # sign flips if phi < 0
                        return fn
                else:
                        return 9999     # if phi does not satisfy the moment inequality

# Note: if phi does not satisfy the moment inequality, fsolve will return phi = 0 

# For each m, find phi that satisfies moment inequality
phi=np.zeros((NS,M))
phi_LB=np.zeros((NS,1))
phi_UB=np.zeros((NS,1))
for i in range(NS):
        for m in range(M):
                v1=offeq_v[i,:,m]
                v0=eq_v[i,:]
                phi_init=0
                phi[i,m]=fsolve(objfn,phi_init)
        phi_LB[i]=np.min(phi[i,:])
        phi_UB[i]=np.max(phi[i,:])

# Note: if phi = 0 then phi is undefined as no exit is observed in equilibrium play

# (3e) Store E[V(s)] using phi_UB, for all s
EVs=np.zeros((NS,1))
for i in range(NS):
        phi=phi_UB[i]
        if phi==100:
                EVs[i]=eq_v[i,0]+eq_v[i,1]*0 # no exit, so no scrap value
        else:
                EVs[i]=eq_v[i,0]+eq_v[i,1]*phi


# (3f) Expected value of an entrant

# Unique states for entrants
se=S[168:len(S),:]
NSe=len(se)

# Expected value of entrant for each unique (0,s2,s3)
EV_entrant=np.zeros((NSe,1))
for s in range(NSe):
        w1=s+28*0
        w2=s+28*1
        w3=s+28*2
        w4=s+28*3
        w5=s+28*4
        w6=s+28*5
        EV_entrant[s]=np.mean((EVs[w1],EVs[w2],EVs[w3],EVs[w4],EVs[w5],EVs[w6]))

# (3g) Solve for kappa (mean and sd)

# Estimated p_enter for each entrant state
p_enter_est=np.zeros((NSe,1))
for s in range(NSe):
        x1=se[s,0]
        x2=se[s,1]
        x3=se[s,2]
        p_enter_est[s]=p_enter(x1,x2,x3)

# Function for p_enter_k
def p_enter_k(kappa):
        mu=kappa[0]
        sigma=kappa[1]
        p_enter_k=np.zeros((NSe,1))
        for s in range(NSe):
                beta=0.95
                x=beta*EV_entrant[s]
                p_enter_k[s]=norm.cdf(x,loc=mu,scale=sigma)
        return p_enter_k

# Function for min dist between p_enter_est and p_enter_hat
def objfn_k(kappa):
        fn=np.linalg.norm(p_enter_est-p_enter_k(kappa))
        return fn

# Minimize
kappa_init=np.array([[0],[1]])
results=minimize(objfn_k,kappa_init,\
        method='Nelder-Mead',options={'maxiter':1000})

 # final_simplex: (array([[ 63.81615668, 127.73173269],
 #       [ 63.81619228, 127.73181503],
 #       [ 63.81619393, 127.73179742]]), array([0.51653053, 0.51653053, 0.51653053]))
 #           fun: 0.5165305280563075
 #       message: 'Optimization terminated successfully.'
 #          nfev: 410
 #           nit: 215
 #        status: 0
 #       success: True
 #             x: array([ 63.81615668, 127.73173269])


# (4) Counterfactuals

# (4a) Use (3f) and (3g) to calculate p_enter
mu_k=results.x[0]
sigma_k=results.x[1]
p_enter_k=np.zeros((NSe,1))
for s in range(NSe):
        p_enter_k[s]=norm.cdf(beta*EV_entrant[s],loc=mu_k,scale=sigma_k)

# (4b) Calculate long-run expected discounted profits,
# entry rates, and markups from paths starting at s0 = (0,3,3) (index: 186)

LR_exp_profits=v_eq[186,0] # LR expected profits
entry_rate_est=p_enter(0,3,3)
entry_rate_hat=p_enter_k[19]
cost=np.zeros((6))
for c in range(6):
        cost[c]=alpha_hat/(alpha_hat+c)

exp_markup=np.mean((pi(1,3,3)/cost[0],pi(2,3,3)/cost[1],pi(3,3,3)/cost[2],\
        pi(4,3,3)/cost[3],pi(5,3,3)/cost[4],pi(6,3,3)/cost[5]))

# (4c) Use (3f) and (3g) to simulate p_enter_1
p_enter_k1=np.zeros((NSe,1))
for s in range(NSe):
        p_enter_k1[s]=norm.cdf(beta*EV_entrant[s],loc=2*mu_k,scale=sigma_k)

# (4d) Use p_enter_k1 to recalculate expected value of entrant
   

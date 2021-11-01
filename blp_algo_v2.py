import numpy as np
import scipy as sp
import pandas as pd
import warnings 
from scipy.optimize import minimize
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.chdir("/Users/christophersaw/Desktop/blp")
data=pd.read_csv(r'headache.csv')
data=data.sort_values(by=['store','week','brand'])

# PART ONE: DEMAND ESTIMATION

# No. of individuals (simulations), products (brands) and markets (store-weeks)
ns=20
nj=11
nt=3504

# PREPARE DATA
# Market shares (nj x nt)
shares=pd.DataFrame(data['shares']).join(data['market']).join(data['brand']).pivot(index='brand',columns='market',values='shares').to_numpy()
outsideshare=pd.DataFrame(data['outsideshare']).join(data['market']).join(data['brand']).pivot(index='brand',columns='market',values='outsideshare').to_numpy()

# Prices (nj x nt)
price=pd.DataFrame(data['price']).join(data['market']).join(data['brand']).pivot(index='brand',columns='market',values='price').to_numpy()

# Branded Product dummy
branded=pd.DataFrame(data['branded_product']).join(data['market']).join(data['brand']).pivot(index='brand',columns='market',values='branded_product').to_numpy()

# Income (nj x nt x ns)
d=pd.read_csv(r'OTCDemographics.csv',sep='\t').sort_values(by=['store','week'])
del(d['store'],d['week'])
d=d.to_numpy()
inc=np.array([np.repeat(i, nj).reshape(ns, nj) for i in d]).swapaxes(1,2).swapaxes(0,1)
inc_jt=pd.DataFrame(inc[:,153,:]) # incomes for store 9 week 10

# Nu (nj x nt x ns)
v=np.random.normal(0,1,(nt,ns))
v=pd.DataFrame(v)
v.columns=["rand_"+str(i) for i in range(1,21)]
v=v.to_numpy()
nu=np.array([np.repeat(i, nj).reshape(ns, nj) for i in v]).swapaxes(1,2).swapaxes(0,1)
nu_jt=pd.DataFrame(nu[:,153,:]) # shocks for store 9 week 10

# SET UP VARIABLES
# X1: Linear parameters = prom + 11 brand dummies (nj*nt x 13)
X1=pd.get_dummies(data['brand'])
X1.columns=["brand_"+str(i) for i in range(1,12)]
X1=X1.join(data['prom']).join(data['price']).to_numpy()

# X2: Non linear parameters price + branded_product (nj*nt x 2)
X2=pd.DataFrame(data['price']).join(data['branded_product']).to_numpy()

# Instruments (nj*nt x 45)
iv=pd.read_csv(r'headache_instr.csv').sort_values(by=['store','week','brand'])
del(iv['store'],iv['week'],iv['brand'],iv['weight'])
# list(iv)
# ['cost', 'avoutprice', 
# 'pricestore1', 'pricestore2', 'pricestore3', 'pricestore4', 'pricestore5', 
# 'pricestore6', 'pricestore7', 'pricestore8', 'pricestore9', 'pricestore10', 
# 'pricestore11', 'pricestore12', 'pricestore13', 'pricestore14', 'pricestore15', 
# 'pricestore16', 'pricestore17', 'pricestore18', 'pricestore19', 'pricestore20', 
# 'pricestore21', 'pricestore22', 'pricestore23', 'pricestore24', 'pricestore25', 
# 'pricestore26', 'pricestore27', 'pricestore28', 'pricestore29', 'pricestore30']
Z=iv.to_numpy()

# Weighting matrix [Z'Z]^(-1)
omega = np.linalg.inv( Z.transpose() @ Z )

# Calculate mean utility, for any given theta2
def delta(theta2):
    # Set problem parameters
    x=0
    maxiter=1000
    diff=1
    epsilon=1e-05
    # Define arrays
    num=np.zeros((nj,nt,ns))
    den=np.zeros((nj,nt,ns)) # take average of num/den along axis 2 to calculate integral numerically
    w_h=np.zeros((nj,nt,maxiter))
    w_h[:,:,0]=np.exp(np.log(shares)-np.log(outsideshare)) # w_init = exp(delta_init); delta_init = log(shares)-log(outsideshare) from logit model
    while diff > epsilon and x < maxiter:
        delta_h=np.log(w_h[:,:,x])
        for i in range(ns):
            num[:,:,i]=np.exp(delta_h[:,:]+theta2[0]*nu[:,:,i]*branded[:,:]+theta2[1]*inc[:,:,i]*price[:,:])
            den[:,:,i]=np.sum(num[j,:,i] for j in range(nj)) + 1
        predicted_shares=np.mean(np.divide(num,den),axis=2)
        w_h[:,:,x+1]=np.multiply(w_h[:,:,x],np.divide(shares,predicted_shares)) # w_h+1 = w_h * (shares/predicted_shares), where w = exp(delta)
        diff=np.linalg.norm(shares - predicted_shares)
        x=x+1
    return np.log(w_h[:,:,x]).flatten('F').transpose()

# GMM objective function
def gmmobjfn(theta2):
    gmmobjfn=(delta(theta2) - X1 @ (np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta(theta2))).transpose() @ Z @ omega @ Z.transpose() @(delta(theta2) - X1 @ (np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta(theta2)))
    return gmmobjfn
# Notes: for given theta2, delta(theta2) returns the mean utilities "delta". From there, we:
# (1) Recover theta1 (linear parameters): theta1 = np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta
# (2) Recover ksi_j (structural residuals): ksi=delta - X1 @ theta1
# The GMM objective function is "ksi.transpose() @ Z @ omega @ Z.transpose() @ksi"; use (1) and (2) and we get the mess above.

# Set intital guess and solve
theta2=np.array([[0.001],       #sigma_b
                 [0.001]])      #sigma_i
results=sp.optimize.minimize(gmmobjfn,theta2,method='Nelder-Mead')
# results
# final_simplex: (array([[0.08765094, 0.05278045],
#       [0.08766798, 0.05278336],
#       [0.08755854, 0.05278304]]), array([11.56531214, 11.56531214, 11.56531214]))
#           fun: 11.5653121401884
#       message: 'Optimization terminated successfully.'
#          nfev: 124
#           nit: 64
#        status: 0
#       success: True
#             x: array([0.08765094, 0.05278045])
theta2_soln=results.x
theta1_soln=np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta
pd.DataFrame(theta1_soln).to_csv('theta1.csv',index=False)
pd.DataFrame(theta2_soln).to_csv('theta2.csv',index=False)

# PART TWO: MARKET ANALYSIS

# Calculate mean utility given solution for theta2
def delta(theta2):
    # Set problem parameters
    x=0
    maxiter=1000
    diff=1
    epsilon=1e-05
    # Define arrays
    num=np.zeros((nj,nt,ns))
    den=np.zeros((nj,nt,ns)) # take average of num/den along axis 2 to calculate integral numerically
    w_h=np.zeros((nj,nt,maxiter))
    w_h[:,:,0]=np.exp(np.log(shares)-np.log(outsideshare)) # w_init = exp(delta_init); delta_init = log(shares)-log(outsideshare) from logit model
    while diff > epsilon and x < maxiter:
        delta_h=np.log(w_h[:,:,x])
        for i in range(ns):
            num[:,:,i]=np.exp(delta_h[:,:]+theta2[0]*nu[:,:,i]*branded[:,:]+theta2[1]*inc[:,:,i]*price[:,:])
            den[:,:,i]=np.sum(num[j,:,i] for j in range(nj)) + 1
        predicted_shares=np.mean(np.divide(num,den),axis=2)
        w_h[:,:,x+1]=np.multiply(w_h[:,:,x],np.divide(shares,predicted_shares)) # w_h+1 = w_h * (shares/predicted_shares), where w = exp(delta)
        diff=np.linalg.norm(shares - predicted_shares)
        x=x+1
    return np.log(w_h[:,:,x]).flatten('F').transpose()

data['delta_jt']=delta(theta2_soln)

# Extract data for store 9, week 10
data2=data.loc[data['market_ids']=='9x10']
branded=pd.DataFrame(data2['branded_product']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='branded_product').to_numpy()
price=pd.DataFrame(data2['price']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='price').to_numpy()
shares=pd.DataFrame(data2['shares']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='shares').to_numpy()
inc=inc_jt.loc[0]
nu=nu_jt.loc[0]

# mu_ij (i's deviation from mean utility of product j)
mu=np.zeros((20,11))
for i in range(20):
    for j in range(11):
        mu[i,j]=theta2_soln[0]*nu[i]*branded[j] + theta2_soln[1]*inc[i]*price[j]

# delta_j (mean utility of product j)
delta=data2['delta_jt'].to_numpy()

# prob_ij (probability of individual i purchasing product j)
num=np.zeros((20,11))
den=np.zeros((20,11))
for i in range(20):
    for j in range(11):
        num[i,:]=np.exp(delta+mu[i,:])
        den[i,:]=np.sum(num[i,k] for k in range(11)) + 1
    prob=np.divide(num,den)

# alpha
alpha=theta1_soln[12]+theta2_soln[1]*inc
alpha=alpha.to_numpy()
alpha_logit=theta1_soln[12]

# Cross- and own- elasticities from Random Coefficients Model
eta=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        if j==k:
            eta[j,j] = (-1)*(price[j]/shares[j])*(np.mean(alpha*prob[:,j]*(1-prob[:,j])))
        else:
            eta[j,k] = (price[k]/shares[j])*(np.mean(alpha*prob[:,j]*prob[:,k]))

elasticities=pd.DataFrame(eta)
elasticities.round(4).to_csv('elasticities_rc.csv')

# Cross- and own- elasticities from Logit Model
eta_logit=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        if j==k:
            eta_logit[j,j] = alpha_logit*price[j]*(1-shares[j])
        else:
            eta_logit[j,k] = (-1)*alpha_logit*price[k]*(shares[k])

elasticities_logit=pd.DataFrame(eta_logit)
elasticities_logit.round(4).to_csv('elasticities_logit.csv')
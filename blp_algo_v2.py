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

# No. of individuals (simulations), products (brands) and markets (store-weeks)
ns=20
nj=11
nt=3504

# PREPARE DATA
# Market shares (nj x nt)
shares=pd.DataFrame(data['shares'])
shares=shares.join(data['market'])
shares=shares.join(data['brand'])
shares=shares.pivot(index='brand',columns='market',values='shares')
shares=shares.to_numpy()

outsideshare=pd.DataFrame(data['outsideshare'])
outsideshare=outsideshare.join(data['market'])
outsideshare=outsideshare.join(data['brand'])
outsideshare=outsideshare.pivot(index='brand',columns='market',values='outsideshare')
outsideshare=outsideshare.to_numpy()

# Prices (nj x nt)
price=pd.DataFrame(data['price'])
price=price.join(data['market'])
price=price.join(data['brand'])
price=price.pivot(index='brand',columns='market',values='price')
price=price.to_numpy()

# Income (nj x nt x ns)
d=pd.read_csv(r'OTCDemographics.csv',sep='\t')
d=d.sort_values(by=['store','week'])
del(d['store'],d['week'])
d=d.to_numpy()
inc=np.array([np.repeat(i, nj).reshape(ns, nj) for i in d]).swapaxes(1,2).swapaxes(0,1)

# Nu (nj x nt x ns)
v=np.random.normal(0,1,(nt,ns))
v=pd.DataFrame(v)
v.columns=["rand_"+str(i) for i in range(1,21)]
v=v.to_numpy()
nu=np.array([np.repeat(i, nj).reshape(ns, nj) for i in v]).swapaxes(1,2).swapaxes(0,1)

# SET UP VARIABLES
# X1: Linear parameters = prom + 11 brand dummies (nj*nt x 13)
x1=pd.get_dummies(data['brand'])
x1.columns=["brand_"+str(i) for i in range(1,12)]
x1=x1.join(data['prom'])
x1=x1.join(data['price'])
X1=x1.to_numpy()

# X2: Non linear parameters price + branded_product (nj*nt x 2)
x2=pd.DataFrame(data['price'])
x2=x2.join(data['branded_product'])
X2=x2.to_numpy()

# Instruments (nj*nt x 45)
iv=pd.read_csv(r'headache_instr.csv')
iv=iv.sort_values(by=['store','week','brand'])
del(iv['store'],iv['week'],iv['brand'],iv['weight'])
iv=pd.merge(iv, x1,left_index=True,right_index=True)
del(iv['price'])
iv=pd.merge(iv, x2,left_index=True,right_index=True)
del(iv['price'])
# list(iv)
# ['cost', 'avoutprice', 
# 'pricestore1', 'pricestore2', 'pricestore3', 'pricestore4', 'pricestore5', 
# 'pricestore6', 'pricestore7', 'pricestore8', 'pricestore9', 'pricestore10', 
# 'pricestore11', 'pricestore12', 'pricestore13', 'pricestore14', 'pricestore15', 
# 'pricestore16', 'pricestore17', 'pricestore18', 'pricestore19', 'pricestore20', 
# 'pricestore21', 'pricestore22', 'pricestore23', 'pricestore24', 'pricestore25', 
# 'pricestore26', 'pricestore27', 'pricestore28', 'pricestore29', 'pricestore30', 
# 'brand_1', 'brand_2', 'brand_3', 'brand_4', 'brand_5', 
# 'brand_6', 'brand_7', 'brand_8', 'brand_9', 'brand_10', 'brand_11', 
# 'prom', 'branded_product']
Z=iv.to_numpy()

# Weighting matrix [Z'Z]^(-1)
omega = np.linalg.inv( Z.transpose() @ Z )

# Contraction mapping
def delta(theta2):
    # Build arrays
    predicted_shares=np.zeros((nj,nt))
    num=np.zeros((nj,nt,ns))
    den=np.zeros((nj,nt,ns)) # take average of num/den along axis 2 to calculate integral numerically
    # Contraction mapping to get deltas
    maxiter=100
    diff=1
    tol=1e-06
    w_h=np.zeros((nj,nt,maxiter))
    w_h[:,:,0]=np.exp(np.log(shares)-np.log(outsideshare)) # Initial delta are values that solve the logit model
    x=0
    while diff > tol and x < maxiter:
        delta_h=np.log(w_h[:,:,x])
        for i in range(ns):
            num[:,:,i]=np.exp(delta_h[:,:]+theta2[0]*nu[:,:,i]+theta2[1]*inc[:,:,i]*price[:,:])
            den[:,:,i]=np.sum(num[j,:,i] for j in range(nj)) +1
        predicted_shares=np.mean(np.divide(num,den),axis=2)
        w_h[:,:,x+1]=np.multiply(w_h[:,:,x],np.divide(shares,predicted_shares))
        diff=np.linalg.norm(shares - predicted_shares)
        x=x+1
    return np.log(w_h[:,:,x]).flatten('F').transpose()

def gmmobjfn(theta2):
    gmmobjfn=(delta(theta2) - X1 @ (np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta(theta2))).transpose() @ Z @ omega @ Z.transpose() @(delta(theta2) - X1 @ (np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta(theta2)))
    return gmmobjfn

theta2=np.array([[1],       #sigma_b
                 [1]])      #sigma_i

results=sp.optimize.minimize(gmmobjfn,theta2,method='Nelder-Mead',return_all=True)

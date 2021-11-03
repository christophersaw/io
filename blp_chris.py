import os
os.chdir("/Users/christophersaw/Desktop/blp")
import pandas as pd
import numpy as np
import linearmodels
from linearmodels import OLS
from linearmodels.iv import IV2SLS
from pystout import pystout
import scipy as sp
from scipy.optimize import minimize
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

### PREPARE DATA ###

# Read data
df=pd.read_csv(r'OTC_Data.csv',sep='\t')
df.columns=df.columns.str.replace('_','')

# Re-weight sales, costs and prices to 50tab package
df.loc[df['brand'].isin([1,4,7]),'weight']=0.5
df.loc[df['brand'].isin([2,5,8,10]),'weight']=1
df.loc[df['brand'].isin([3,6,9,11]),'weight']=2
df['sales']	= df['sales']*df['weight']
df['price']	= df['price']/df['weight']
df['cost']	= df['cost']/df['weight']

# Create categories/dummies for brand, store-brand and branded_product
df['branded_product']=0
df.loc[df['brand'].isin([1,2,3,4,5,6,7,8,9]), 'branded_product']=1

# Market ids, market shares
df['market']=df.groupby(['store','week']).ngroup()
df['market_ids']=df['store'].astype(str)+str('x')+df['week'].astype(str)
df['shares']=df['sales']/df['count']

# Calculate inside and outside shares
df['insideshare']=df.groupby(['market'])['shares'].transform('sum')
df['outsideshare']=df['insideshare'].apply(lambda x: 1-x)

# Save data for BLP model
df.sort_values(by=['store','week']).to_csv('headache.csv',index=False)

# Instruments
df2=pd.read_csv(r'OTCDataInstruments.csv',sep='\t')
df2=df2.sort_values(by=['store','week','brand'])

# Re-weight IVs
df2.columns=df2.columns.str.replace('_','')
df2.loc[df2['brand'].isin([1,4,7]),'weight']=0.5
df2.loc[df2['brand'].isin([2,5,8,10]),'weight']=1
df2.loc[df2['brand'].isin([3,6,9,11]),'weight']=2
df2['cost']=df2['cost']/df2['weight']
df2['avoutprice']=df2['avoutprice']/df2['weight']

for store in range(1, 31):
	df2['pricestore'+str(store)] = df2['pricestore'+str(store)]/df2['weight']

# Save instruments for BLP model
df2.sort_values(by=['store','week']).to_csv('headache_instr.csv',index=False)

### Q1: OLS/IV REGRESSIONS FOR LOGIT ###

# Load data for regressions
data=pd.read_csv(r'headache.csv')

# Create dependent variable: Y = log(s_jt) - log(s_0t)
data['log_share']=data['shares'].apply(lambda x: np.log(x))
data['log_outside']=data['outsideshare'].apply(lambda x: np.log(x))
data['y']=data['log_share']-data['log_outside']

# Create categorical variables
data['brand_dummies']=data['brand'].astype('category')
data['store_brand_dummies']=data.groupby(['store','brand']).ngroup().astype('category')

# Create average hausman price instrument
data['total_price']=data.groupby(['brand'])['price'].transform('sum')
data['price_excl_own']=data['total_price'] - data['price']
data['avg_h_price']=data['price_excl_own'].apply(lambda x: x/3503)

# Regressions for Q1.1 to Q1.5
exog_vars1=['price','prom']
X1=data[exog_vars1]
model1 = OLS(data.y, X1).fit()

exog_vars2=['price','prom','brand_dummies']
X2=data[exog_vars2]
model2 = OLS(data.y, X2).fit()

exog_vars3=['price','prom','store_brand_dummies']
X3=data[exog_vars3]
model3=OLS(data.y, X3).fit()

exog_vars4=['prom']
X4=data[exog_vars4]
model4=IV2SLS(data.y, X4, data.price, data.cost).fit()

exog_vars5=['prom','brand_dummies']
X5=data[exog_vars5]
model5=IV2SLS(data.y, X5, data.price, data.cost).fit()

exog_vars6=['prom','store_brand_dummies']
X6=data[exog_vars6]
model6=IV2SLS(data.y, X6, data.price, data.cost).fit()

exog_vars7=['prom']
X7=data[exog_vars7]
model7=IV2SLS(data.y, X7, data.price, data.avg_h_price).fit()

exog_vars8=['prom','brand_dummies']
X8=data[exog_vars8]
model8=IV2SLS(data.y, X8, data.price, data.avg_h_price).fit()

exog_vars9=['prom','store_brand_dummies']
X9=data[exog_vars9]
model9=IV2SLS(data.y, X9, data.price, data.avg_h_price).fit()


# Save results
pystout(models=[model1,model2,model3],
        file='LogitOLSregressions.tex',
        digits=3,
        mgroups={'Logit (OLS)':[1,3]},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        )

pystout(models=[model4,model5,model6],
        file='LogitIVCostregressions.tex',
        digits=3,
        mgroups={'Logit (IV-Cost)':[1,3]},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        )

pystout(models=[model7,model8,model9],
        file='LogitIVHausmanregressions.tex',
        digits=3,
        mgroups={'Logit (IV-Hausman)':[1,3]},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        )

pystout(models=[model1,model2,model3,model4,model5,model6,model7,model8,model9],
        file='LogitAllregressionsv2.tex',
        digits=2,
        mgroups={'Logit (OLS)':[1,3],'Logit (IV - Cost)':[4,6],'Logit (IV - Hausman)':[7,9]},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'},
        addnotes=['Standard errors in parentheses','\sym{**} p\sym{_<}0.01, * p\sym{_<}0.05']
        )

# Calculate average own-price elasticities from models 1 to 9
data['alpha1']=model1.params[0]
data['alpha2']=model2.params[0]
data['alpha3']=model3.params[0]
data['eta1']=data['alpha1']*data['price']*(1-data['shares'])
data['eta2']=data['alpha2']*data['price']*(1-data['shares'])
data['eta3']=data['alpha3']*data['price']*(1-data['shares'])
data['eta1'].mean() # -6.72
data['eta2'].mean() # -4.68
data['eta3'].mean() # -2.20

### Q2: BLP ###

# Reload data and sort by market, product
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

# Branded Product dummy (nj x nt)
branded=pd.DataFrame(data['branded_product']).join(data['market']).join(data['brand']).pivot(index='brand',columns='market',values='branded_product').to_numpy()

# Income (nj x nt x ns)
d=pd.read_csv(r'OTCDemographics.csv',sep='\t').sort_values(by=['store','week'])
del(d['store'],d['week'])
d=d.to_numpy()
inc=np.array([np.repeat(i, nj).reshape(ns, nj) for i in d]).swapaxes(1,2).swapaxes(0,1)
inc_t=pd.DataFrame(inc[:,153,:]) # incomes for store 9 week 10

# Nu (nj x nt x ns)
v=np.random.normal(0,1,(nt,ns))
v=pd.DataFrame(v)
v.columns=["rand_"+str(i) for i in range(1,21)]
v=v.to_numpy()
nu=np.array([np.repeat(i, nj).reshape(ns, nj) for i in v]).swapaxes(1,2).swapaxes(0,1)
nu_t=pd.DataFrame(nu[:,153,:]) # shocks for store 9 week 10

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
    theta1=np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta(theta2) # (1) Recover theta1 (linear parameters)
    xi=delta(theta2) - X1 @ theta1 # (2) Recover xi_j (structural residuals)
    gmmobjfn=xi.transpose() @ Z @ omega @ Z.transpose() @xi # GMM objective function
    return gmmobjfn

# SOLUTION
# Set intital guess and solve
theta2=np.array([[0.001],       #sigma_b
                 [0.001]])      #sigma_i
results=sp.optimize.minimize(gmmobjfn,theta2,method='Nelder-Mead')
# results
# final_simplex: (array([[1.0087855 , 0.05866099],
#       [1.00886377, 0.05866039],
#       [1.008784  , 0.058657  ]]), array([8.89916613, 8.89916614, 8.89916614]))
#           fun: 8.899166132258753
#       message: 'Optimization terminated successfully.'
#          nfev: 127
#           nit: 67
#        status: 0
#       success: True
#             x: array([1.0087855 , 0.05866099])
theta2_soln=results.x
delta=delta(theta2_soln)
theta1_soln=np.linalg.inv(X1.transpose() @ Z @ omega @ Z.transpose() @ X1) @ X1.transpose() @ Z @ omega @ Z.transpose() @ delta
theta1=pd.DataFrame(theta1_soln)
theta1.columns=['X1']
list1=["brand_"+str(i) for i in range(1,12)]
list2=['prom','price']
theta1.index=list1+list2
theta1.to_csv('theta1.csv')

theta2=pd.DataFrame(theta2_soln)
theta2.columns=['X2']
theta2.index=['sigma_b','sigma_i']
theta2.to_csv('theta2.csv')

gmmval=results.fun

# PART TWO: MARKET ANALYSIS

# Mean utility given solution for theta2 (all variables below have subscript t=9x10 omitted)
data['delta_t']=delta

# Store 9 Week 10
data2=data.loc[data['market_ids']=='9x10']
branded=pd.DataFrame(data2['branded_product']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='branded_product').to_numpy()
price=pd.DataFrame(data2['price']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='price').to_numpy()
shares=pd.DataFrame(data2['shares']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='shares').to_numpy()
inc=inc_t.loc[0]
nu=nu_t.loc[0]

# mu_ij (i's deviation from mean utility of product j)
mu=np.zeros((20,11))
for i in range(20):
    for j in range(11):
        mu[i,j]=theta2_soln[0]*nu[i]*branded[j] + theta2_soln[1]*inc[i]*price[j]

# delta_j (mean utility of product j)
delta=data2['delta_t'].to_numpy()

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
brandlist= ['Tylenol (25)',
            'Tylenol (50)',
            'Tylenol (100)',
            'Advil (25)',
            'Advil (50)',
            'Advil (100)',
            'Bayer (25)',
            'Bayer (50)',
            'Bayer (100)',
            'Store Brand (50)',
            'Store Brand (100)']
elasticities.columns=brandlist
elasticities.index=brandlist
elasticities.round(6).to_csv('elasticities_rc.csv')

# Cross- and own- elasticities from Logit Model
eta_logit=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        if j==k:
            eta_logit[j,j] = alpha_logit*price[j]*(1-shares[j])
        else:
            eta_logit[j,k] = (-1)*alpha_logit*price[k]*(shares[k])

elasticities_logit=pd.DataFrame(eta_logit)
elasticities_logit.columns=brandlist
elasticities_logit.index=brandlist
elasticities_logit.round(6).to_csv('elasticities_logit.csv')


# PART THREE: MARGINAL COSTS

# Ownership matrix: element (j,k) = 1 if j and k belong to the same company
# For this part assume that each brand is owned by a single company (i.e. 11 brands = 11 companies)
own=np.diag(np.ones(11))

# Cross-derivatives from random coefficients model: eta_jk = (ds_j/dp_k)*(p_k/s_j)
cross=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        cross[j,k]=eta[j,k]*shares[j]*price[k]

# Derive marginal costs from FOC
marginalcosts=price + np.linalg.inv(np.multiply(own,cross)) @ shares
marginalcosts=pd.DataFrame(marginalcosts)

# Compare with wholesale costs (adjusted for quantitites)
wholesalecosts=data2['cost'].reset_index()
del(wholesalecosts['index'])
compare=marginalcosts.join(wholesalecosts)
compare.columns=['Marginal Costs','Wholesale Costs']
compare.index=brandlist
compare.round(4).to_csv("cost_comparison.csv")


### Q3: MERGER ANALYSIS ### 

# # Ownership matrix: element (j,k) = 1 if j and k belong to the same company
# For this part assume that:
# Pre-merger: Tylenol owns brands 1 to 3, Advil owns brands 4 to 6, and Bayer owns brands 7 to 9
# Post-merger: one company owns brands 1 to 9
own_pre=np.zeros((11,11))
for i in range(0,3):
    for j in range(0+(3*i),3+(3*i)):
        for k in range(0+(3*i),3+(3*i)):
            own_pre[j,k]=1

for i in range(9,11):
    own_pre[i,i]=1

own_post=np.zeros((11,11))
for j in range(0,9):
    for k in range(0,9):
        own_post[j,k]=1

for i in range(9,11):
    own_post[i,i]=1

# Cross-derivatives from logit model: eta_logit_jk = (ds_j/dp_k)*(p_k/s_j)
cross_logit=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        cross_logit[j,k]=eta_logit[j,k]*shares[j]*price[k]

# Derive marginal costs from FOC
marginalcosts_pre=price + np.linalg.inv(np.multiply(own_pre,cross_logit)) @ shares

# Predict post-merger prices using post-merger ownership matrix
price_post=marginalcosts_pre - np.linalg.inv(np.multiply(own_post,cross_logit)) @ shares

# Save results
price_increase=np.divide(price_post,price)
price_increase=pd.DataFrame(price_increase)
price_post=pd.DataFrame(price_post).merge(price_increase, left_index=True, right_index=True)
price=pd.DataFrame(price).merge(price_post, left_index=True, right_index=True)
price.columns=['Pre-merger price','Post-merger price', 'Percentage Change']
price['Percentage Change']=(price['Percentage Change']-1)*100
price.index=brandlist
price.round(6).to_csv('predicted_prices_logit.csv')

# Reload prices and shares
price=pd.DataFrame(data2['price']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='price').to_numpy()
shares=pd.DataFrame(data2['shares']).join(data2['market']).join(data2['brand']).pivot(index='brand',columns='market',values='shares').to_numpy()

# Cross-derivatives from random coefficients model: eta_jk = (ds_j/dp_k)*(p_k/s_j)
cross=np.zeros((11,11))
for j in range(11):
    for k in range(11):
        cross[j,k]=eta[j,k]*shares[j]*price[k]

# Derive marginal costs from FOC
marginalcosts_pre=price + np.linalg.inv(np.multiply(own_pre,cross)) @ shares

# Predict post-merger prices using post-merger ownership matrix
price_post=marginalcosts_pre - np.linalg.inv(np.multiply(own_post,cross)) @ shares

# Save results
price_increase=np.divide(price_post,price)
price_increase=pd.DataFrame(price_increase)
price_post=pd.DataFrame(price_post).merge(price_increase, left_index=True, right_index=True)
price=pd.DataFrame(price).merge(price_post, left_index=True, right_index=True)
price.columns=['Pre-merger price','Post-merger price', 'Percentage Change']
price['Percentage Change']=(price['Percentage Change']-1)*100
price.index=brandlist
price.round(6).to_csv('predicted_prices_rc.csv')
import os
import pandas as pd
import numpy as np
import scipy
import linearmodels
from linearmodels import OLS
from linearmodels.iv import IV2SLS
from pystout import pystout

os.chdir("/Users/christophersaw/Desktop/blp")
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

data['alpha4']=model4.params[0]
data['alpha5']=model5.params[0]
data['alpha6']=model6.params[0]

data['alpha7']=model7.params[0]
data['alpha8']=model8.params[0]
data['alpha9']=model9.params[0]

data['eta1']=data['alpha1']*data['price']*(1-data['shares'])
data['eta2']=data['alpha2']*data['price']*(1-data['shares'])
data['eta3']=data['alpha3']*data['price']*(1-data['shares'])

data['eta4']=data['alpha4']*data['price']*(1-data['shares'])
data['eta5']=data['alpha5']*data['price']*(1-data['shares'])
data['eta6']=data['alpha6']*data['price']*(1-data['shares'])

data['eta7']=data['alpha7']*data['price']*(1-data['shares'])
data['eta8']=data['alpha8']*data['price']*(1-data['shares'])
data['eta9']=data['alpha9']*data['price']*(1-data['shares'])

data['eta1'].mean() # -6.72
data['eta2'].mean() # -4.68
data['eta3'].mean() # -2.20

data['eta4'].mean() #
data['eta5'].mean() #
data['eta6'].mean() #

data['eta7'].mean() #
data['eta8'].mean() #
data['eta9'].mean() #
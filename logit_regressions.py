import os
import pandas as pd
import numpy as np
import scipy
import linearmodels
from linearmodels import OLS
from linearmodels.iv import IV2SLS

os.chdir("/Users/christophersaw/Desktop/blp")
data=pd.read_csv(r'headache.csv')

# Create dependent variable: Y = log(s_jt) - log(s_0t)
data['log_share']=data['shares'].apply(lambda x: np.log(x))
data['log_outside']=data['outsideshare'].apply(lambda x: np.log(x))
data['y']=data['log_share']-data['log_outside']

# Create categorical variables
data['brand_dummies']=data['brand'].astype('category')
data['store_brand_dummies']=data['brand']*data['store']
data['store_brand_dummies']=data['store_brand_dummies'].astype('category')

# Create average hausman price instrument
data['total_price']=data.groupby(['brand'])['price'].transform('sum')
data['price_excl_own']=data['total_price'] - data['price']
data['avg_h_price']=data['price_excl_own'].apply(lambda x: x/3503)

# Regressions for Q1.1 to Q1.5
exog_vars=['price','prom']
X1=data[exog_vars]
model1 = OLS(data.y, X1).fit()
print(model1)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8823
# Estimator:                        OLS   Adj. R-squared:                 0.8823
# No. Observations:               38544   F-statistic:                 4.091e+05
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:37:06   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# price         -1.6235     0.0028    -578.07     0.0000     -1.6290     -1.6180
# prom          -2.5092     0.0346    -72.435     0.0000     -2.5771     -2.4413
# ==============================================================================
exog_vars=['price','prom','brand_dummies']
X2=data[exog_vars]
model2 = OLS(data.y, X2).fit()
print(model2)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9912
# Estimator:                        OLS   Adj. R-squared:                 0.9912
# No. Observations:               38544   F-statistic:                 5.153e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:38:37   Distribution:                 chi2(12)
# Cov. Estimator:                robust                                         
                                                                              
#                                 Parameter Estimates                                 
# ====================================================================================
#                   Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------------
# price               -1.1315     0.0014    -820.34     0.0000     -1.1342     -1.1288
# prom                 0.1242     0.0165     7.5116     0.0000      0.0918      0.1566
# [Brand effects: Yes]
# ====================================================================================
exog_vars=['price','prom','store_brand_dummies']
X3=data[exog_vars]
model3=OLS(data.y, X3).fit()
print(model3)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9906
# Estimator:                        OLS   Adj. R-squared:                 0.9904
# No. Observations:               38544   F-statistic:                 7.503e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:39:43   Distribution:                chi2(562)
# Cov. Estimator:                robust                                         
                                                                              
#                                     Parameter Estimates                                     
# ============================================================================================
#                           Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# --------------------------------------------------------------------------------------------
# price                       -0.2547     0.0105    -24.226     0.0000     -0.2753     -0.2341
# prom                         0.2966     0.0153     19.390     0.0000      0.2666      0.3266
# [Store-brand effects: Yes]
# ============================================================================================
exog_vars=['prom']
X1=data[exog_vars]
model4=IV2SLS(data.y, X1, data.price, data.demand_instruments0).fit()
print(model4)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8823
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.8823
# No. Observations:               38544   F-statistic:                 4.511e+05
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:41:29   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# price         -1.6172     0.0027    -609.93     0.0000     -1.6224     -1.6120
# prom          -2.5362     0.0344    -73.686     0.0000     -2.6036     -2.4687
# ==============================================================================

# Endogenous: price
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','brand_dummies']
X2=data[exog_vars]
model5=IV2SLS(data.y, X2, data.price, data.demand_instruments0).fit()
print(model5)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9912
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9912
# No. Observations:               38544   F-statistic:                 5.213e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:41:50   Distribution:                 chi2(12)
# Cov. Estimator:                robust                                         
                                                                              
#                                 Parameter Estimates                                 
# ====================================================================================
#                   Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------------
# price               -1.1376     0.0013    -843.13     0.0000     -1.1402     -1.1349
# prom                 0.1242     0.0165     7.5116     0.0000      0.0918      0.1566
# [Brand effects: Yes]
# ====================================================================================

# Endogenous: price
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','store_brand_dummies']
X3=data[exog_vars]
model6=IV2SLS(data.y, X3, data.price, data.demand_instruments0).fit()
print(model6)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9905
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9903
# No. Observations:               38544   F-statistic:                 7.533e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:43:49   Distribution:                chi2(562)
# Cov. Estimator:                robust                                         
                                                                              
#                                     Parameter Estimates                                     
# ============================================================================================
#                           Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# --------------------------------------------------------------------------------------------
# price                       -0.1636     0.0116    -14.090     0.0000     -0.1864     -0.1409
# prom                         0.3371     0.0154     21.818     0.0000      0.3068      0.3673
# [Store-brand effects: yes]
# ============================================================================================

# Endogenous: price
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom']
X1=data[exog_vars]
model7=IV2SLS(data.y, X1, data.price, data.avg_h_price).fit()
print(model7)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8823
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.8823
# No. Observations:               38544   F-statistic:                 4.064e+05
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:48:07   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# price         -1.6305     0.0028    -575.04     0.0000     -1.6361     -1.6250
# prom          -2.4788     0.0349    -71.053     0.0000     -2.5472     -2.4105
# ==============================================================================

# Endogenous: price
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','brand_dummies']
X2=data[exog_vars]
model8=IV2SLS(data.y, X2, data.price, data.avg_h_price).fit()
print(model8)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9911
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9911
# No. Observations:               38544   F-statistic:                 5.068e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:48:26   Distribution:                 chi2(12)
# Cov. Estimator:                robust                                         
                                                                              
#                                 Parameter Estimates                                 
# ====================================================================================
#                   Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------------
# price               -1.1585     0.0017    -672.87     0.0000     -1.1619     -1.1551
# prom                 0.1240     0.0165     7.5072     0.0000      0.0916      0.1564
# [Brand effects: Yes]
# ====================================================================================

# Endogenous: price
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','store_brand_dummies']
X3=data[exog_vars]
model9=IV2SLS(data.y, X3, data.price, data.avg_h_price).fit()
print(model9)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9906
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9904
# No. Observations:               38544   F-statistic:                 7.499e+06
# Date:                Thu, Oct 28 2021   P-value (F-stat)                0.0000
# Time:                        16:56:32   Distribution:                chi2(562)
# Cov. Estimator:                robust                                         
                                                                              
#                                     Parameter Estimates                                     
# ============================================================================================
#                           Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# --------------------------------------------------------------------------------------------
# price                       -0.2524     0.0121    -20.903     0.0000     -0.2761     -0.2288
# prom                         0.2976     0.0156     19.125     0.0000      0.2671      0.3281
# [Store-brand effects: Yes]
# ============================================================================================

# Endogenous: price
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False


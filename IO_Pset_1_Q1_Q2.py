# Attribution: this code is based on PyBLP by Conlon and Gortmaker (2020)
# and closely follows the syntax in their tutorial
# https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/nevo.html

# PART ONE: 	PREPARE DATA
# PART TWO: 	LOGIT
# PART THREE: 	BLP

### PART ONE
# set path and import data
import os
os.chdir("/Users/christophersaw/Desktop/blp")
import pandas as pd
import numpy as np
import scipy
data=pd.read_csv(r'OTC_Data.csv',sep='\t')
df=pd.DataFrame(data)
df=df.rename(columns={'sales_': 'sales', 'price_': 'price', 'prom_': 'prom','cost_': 'cost', 'brand': 'product'})
# 'brand' is renamed as 'product' because we have 11 products and 3 brands (tylenol, advil, bayer)


# normalise sales price and cost to 50tab 
df['packagesales']=df['sales']
df['packageprice']=df['price']
df['packagecost']=df['cost']
df.loc[df['product'] == 1, 'sales'] = df['packagesales'].apply(lambda x: x*0.5)	# sales of 25tab packages weighted at 0.5
df.loc[df['product'] == 4, 'sales'] = df['packagesales'].apply(lambda x: x*0.5)
df.loc[df['product'] == 7, 'sales'] = df['packagesales'].apply(lambda x: x*0.5)
df.loc[df['product'] == 3, 'sales'] = df['packagesales'].apply(lambda x: x*2)	# sales of 100tab packages weighted at 2
df.loc[df['product'] == 6, 'sales'] = df['packagesales'].apply(lambda x: x*2)
df.loc[df['product'] == 9, 'sales'] = df['packagesales'].apply(lambda x: x*2)
df.loc[df['product'] == 11, 'sales'] = df['packagesales'].apply(lambda x: x*2)
df['price']=df['packageprice']*df['packagesales']/df['sales']	# price is now unit price
df['cost']=df['packagecost']*df['packagesales']/df['sales']		# cost is now unit cost


# create brand dummies for tylenol, advil, bayer
df['tylenol']=0
df.loc[df['product'].isin([1,2,3]), 'tylenol']=1
df['advil']=0
df.loc[df['product'].isin([4,5,6]), 'advil']=1
df['bayer']=0
df.loc[df['product'].isin([7,8,9]), 'bayer']=1


# create brand x store dummies
df['inter1']=df['tylenol']*df['store']
df['inter2']=df['advil']*df['store']
df['inter3']=df['bayer']*df['store']


# market ids and shares
df['market']=df.groupby(['store','week']).ngroup()
df['x']='x'
df['market_ids']=df['store'].astype(str)+df['x'].astype(str)+df['week'].astype(str)
del(df['x'])


# market shares
df['totalsales']=df.groupby(['market'])['sales'].transform('sum')
df['bestweek']=df.groupby(['store'])['totalsales'].transform('max')
df['mktsize']=df['bestweek'].apply(lambda x: x*2)
df['shares']=df['sales']/df['mktsize']

 
# calculate inside and outside shares
df['insideshare']=df.groupby(['market'])['shares'].transform('sum')
# check that insideshares are between 0 and 1
df['insideshare'].min()
df['insideshare'].max()
df['outsideshare']=df['insideshare'].apply(lambda x: 1-x)


# load BLP instruments
iv=pd.read_csv(r'OTCDataInstruments.csv',sep='\t')
df2=pd.DataFrame(iv)
# rename brand as product
# note that cost_ in df2 is packagecost in df1
df2=df2.rename(columns={'brand': 'product'})


# merge data and instruments (demographics come in later)
dfm=pd.merge(df, df2, on=['store','week','product']) # this is a one-one merge
#dfm.sort_values(by=['store','week','product'])
#dfm.to_csv('dfm.csv', index=False)


# hausman price instruments are package prices, convert to unit prices
dfm['pricestore1']=dfm['pricestore1']*dfm['packagesales']/dfm['sales']
dfm['pricestore2']=dfm['pricestore2']*dfm['packagesales']/dfm['sales']
dfm['pricestore3']=dfm['pricestore3']*dfm['packagesales']/dfm['sales']
dfm['pricestore4']=dfm['pricestore4']*dfm['packagesales']/dfm['sales']
dfm['pricestore5']=dfm['pricestore5']*dfm['packagesales']/dfm['sales']
dfm['pricestore6']=dfm['pricestore6']*dfm['packagesales']/dfm['sales']
dfm['pricestore7']=dfm['pricestore7']*dfm['packagesales']/dfm['sales']
dfm['pricestore8']=dfm['pricestore8']*dfm['packagesales']/dfm['sales']
dfm['pricestore9']=dfm['pricestore9']*dfm['packagesales']/dfm['sales']
dfm['pricestore10']=dfm['pricestore10']*dfm['packagesales']/dfm['sales']
dfm['pricestore11']=dfm['pricestore11']*dfm['packagesales']/dfm['sales']
dfm['pricestore12']=dfm['pricestore12']*dfm['packagesales']/dfm['sales']
dfm['pricestore13']=dfm['pricestore13']*dfm['packagesales']/dfm['sales']
dfm['pricestore14']=dfm['pricestore14']*dfm['packagesales']/dfm['sales']
dfm['pricestore15']=dfm['pricestore15']*dfm['packagesales']/dfm['sales']
dfm['pricestore16']=dfm['pricestore16']*dfm['packagesales']/dfm['sales']
dfm['pricestore17']=dfm['pricestore17']*dfm['packagesales']/dfm['sales']
dfm['pricestore18']=dfm['pricestore18']*dfm['packagesales']/dfm['sales']
dfm['pricestore19']=dfm['pricestore19']*dfm['packagesales']/dfm['sales']
dfm['pricestore20']=dfm['pricestore20']*dfm['packagesales']/dfm['sales']
dfm['pricestore21']=dfm['pricestore21']*dfm['packagesales']/dfm['sales']
dfm['pricestore22']=dfm['pricestore22']*dfm['packagesales']/dfm['sales']
dfm['pricestore23']=dfm['pricestore23']*dfm['packagesales']/dfm['sales']
dfm['pricestore24']=dfm['pricestore24']*dfm['packagesales']/dfm['sales']
dfm['pricestore25']=dfm['pricestore25']*dfm['packagesales']/dfm['sales']
dfm['pricestore26']=dfm['pricestore26']*dfm['packagesales']/dfm['sales']
dfm['pricestore27']=dfm['pricestore27']*dfm['packagesales']/dfm['sales']
dfm['pricestore28']=dfm['pricestore28']*dfm['packagesales']/dfm['sales']
dfm['pricestore29']=dfm['pricestore29']*dfm['packagesales']/dfm['sales']
dfm['pricestore30']=dfm['pricestore30']*dfm['packagesales']/dfm['sales']


# rename columns to fit pyblp syntax
dfm=dfm.rename(columns={'price': 'prices'})
dfm=dfm.rename(columns={'cost': 'demand_instruments0',
'pricestore1': 'demand_instruments1','pricestore2': 'demand_instruments2','pricestore3': 'demand_instruments3','pricestore4': 'demand_instruments4',
'pricestore5': 'demand_instruments5','pricestore6': 'demand_instruments6','pricestore7': 'demand_instruments7','pricestore8': 'demand_instruments8',
'pricestore9': 'demand_instruments9','pricestore10': 'demand_instruments10','pricestore11': 'demand_instruments11','pricestore12': 'demand_instruments12',
'pricestore13': 'demand_instruments13','pricestore14': 'demand_instruments14','pricestore15': 'demand_instruments15','pricestore16': 'demand_instruments16',
'pricestore17': 'demand_instruments17','pricestore18': 'demand_instruments18','pricestore19': 'demand_instruments19','pricestore20': 'demand_instruments20',
'pricestore21': 'demand_instruments21','pricestore22': 'demand_instruments22','pricestore23': 'demand_instruments23','pricestore24': 'demand_instruments24',
'pricestore25': 'demand_instruments25','pricestore26': 'demand_instruments26','pricestore27': 'demand_instruments27','pricestore28': 'demand_instruments28',
'pricestore29': 'demand_instruments29','pricestore30': 'demand_instruments30'})


# Load demographic data and reshape for pyblp (each row is an agent in market t)
inc=pd.read_csv(r'OTCDemographics.csv',sep='\t')
df3=pd.DataFrame(inc)
df3['x']='x'
df3['market_ids']=df3['store'].astype(str)+df3['x'].astype(str)+df3['week'].astype(str)
del(df3['x'])
df4=pd.melt(df3, id_vars=['market_ids'], value_vars=[
	'hhincome1','hhincome2','hhincome3','hhincome4','hhincome5','hhincome6','hhincome7','hhincome8','hhincome9','hhincome10',
	'hhincome11','hhincome12','hhincome13','hhincome14','hhincome15','hhincome16','hhincome17','hhincome18','hhincome19','hhincome20'], 
	var_name='agent_index', value_name='income')
del(df4['agent_index'])



### PART TWO: OLS/IV Logit regressions
import linearmodels
from linearmodels import OLS
from linearmodels.iv import IV2SLS


# Create dependent variable: Y = log(s_jt) - log(s_0t)
dfm['log_share']=dfm['shares'].apply(lambda x: np.log(x))
dfm['log_outside']=dfm['outsideshare'].apply(lambda x: np.log(x))
dfm['y']=dfm['log_share']-dfm['log_outside']


# Create store*brand dummies
dfm['inter1']=dfm['inter1'].astype('category')
dfm['inter2']=dfm['inter2'].astype('category')
dfm['inter3']=dfm['inter3'].astype('category')
# set store 9 in week 10 as the base
dfm.loc[dfm['market_ids']=="9x10", 'inter1'] = 0
dfm.loc[dfm['market_ids']=="9x10", 'inter2'] = 0
dfm.loc[dfm['market_ids']=="9x10", 'inter3'] = 0 


# Regressions for Q1.1 to Q1.5
exog_vars=['prices','prom']
X1=dfm[exog_vars]
model1 = OLS(dfm.y, X1).fit()
print(model1)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8488
# Estimator:                        OLS   Adj. R-squared:                 0.8487
# No. Observations:               38544   F-statistic:                 2.708e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        19:59:07   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.7298     0.0015    -473.34     0.0000     -0.7328     -0.7268
# prom          -0.8224     0.0194    -42.390     0.0000     -0.8604     -0.7844
# ==============================================================================
exog_vars=['prices','prom','tylenol','advil','bayer']
X2=dfm[exog_vars]
model2 = OLS(dfm.y, X2).fit()
print(model2)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9016
# Estimator:                        OLS   Adj. R-squared:                 0.9015
# No. Observations:               38544   F-statistic:                 4.905e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        19:59:31   Distribution:                  chi2(5)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8484     0.0049    -173.04     0.0000     -0.8580     -0.8388
# prom          -0.3279     0.0190    -17.271     0.0000     -0.3651     -0.2907
# tylenol        1.5197     0.0267     56.879     0.0000      1.4673      1.5721
# advil          0.6427     0.0268     23.964     0.0000      0.5902      0.6953
# bayer         -0.8569     0.0194    -44.104     0.0000     -0.8950     -0.8188
# ==============================================================================
exog_vars=['prices','prom','inter1','inter2','inter3']
X3=dfm[exog_vars]
model3=OLS(dfm.y, X3).fit()
print(model3)
#                             OLS Estimation Summary                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9048
# Estimator:                        OLS   Adj. R-squared:                 0.9042
# No. Observations:               38544   F-statistic:                 5.571e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:14:16   Distribution:                chi2(221)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8508     0.0049    -175.30     0.0000     -0.8603     -0.8412
# prom          -0.3276     0.0187    -17.557     0.0000     -0.3641     -0.2910
# (omitted)
# ==============================================================================
exog_vars=['prom']
X1=dfm[exog_vars]
model4=IV2SLS(dfm.y, X1, dfm.prices, dfm.demand_instruments0).fit()
print(model4)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8487
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.8487
# No. Observations:               38544   F-statistic:                 2.893e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:00:38   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.7226     0.0015    -489.79     0.0000     -0.7255     -0.7197
# prom          -0.8534     0.0192    -44.333     0.0000     -0.8911     -0.8156
# ==============================================================================

# Endogenous: prices
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','tylenol','advil','bayer']
X2=dfm[exog_vars]
model5=IV2SLS(dfm.y, X2, dfm.prices, dfm.demand_instruments0).fit()
print(model5)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9012
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9012
# No. Observations:               38544   F-statistic:                  5.29e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:01:28   Distribution:                  chi2(5)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8032     0.0049    -162.48     0.0000     -0.8129     -0.7935
# prom          -0.3292     0.0192    -17.188     0.0000     -0.3668     -0.2917
# tylenol        1.2894     0.0269     47.888     0.0000      1.2366      1.3421
# advil          0.4145     0.0271     15.284     0.0000      0.3614      0.4677
# bayer         -1.0214     0.0197    -51.912     0.0000     -1.0600     -0.9829
# ==============================================================================

# Endogenous: prices
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
dfm['inter1']=dfm['inter1'].astype('category')
dfm['inter2']=dfm['inter2'].astype('category')
dfm['inter3']=dfm['inter3'].astype('category')
# set store 9 in week 10 as the base
dfm.loc[dfm['market_ids']=="9x10", 'inter1'] = 0
dfm.loc[dfm['market_ids']=="9x10", 'inter2'] = 0
dfm.loc[dfm['market_ids']=="9x10", 'inter3'] = 0 

exog_vars=['prom','inter1','inter2','inter3']
X3=dfm[exog_vars]
model6=IV2SLS(dfm.y, X3, dfm.prices, dfm.demand_instruments0).fit()
print(model6)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9044
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9039
# No. Observations:               38544   F-statistic:                 6.079e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:02:34   Distribution:                chi2(221)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8037     0.0049    -164.67     0.0000     -0.8133     -0.7941
# prom          -0.3290     0.0188    -17.462     0.0000     -0.3660     -0.2921
# (omitted)
# ==============================================================================

# Endogenous: prices
# Instruments: demand_instruments0
# Robust Covariance (Heteroskedastic)
# Debiased: False
dfm['total_price']=dfm.groupby(['product'])['prices'].transform('sum')
dfm['price_excl_own']=dfm['total_price'] - dfm['prices']
dfm['avg_h_price']=dfm['price_excl_own'].apply(lambda x: x/3503)

exog_vars=['prom']
X1=dfm[exog_vars]
model7=IV2SLS(dfm.y, X1, dfm.prices, dfm.avg_h_price).fit()
print(model7)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.8487
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.8487
# No. Observations:               38544   F-statistic:                 2.657e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:04:36   Distribution:                  chi2(2)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.7323     0.0016    -468.60     0.0000     -0.7354     -0.7293
# prom          -0.8114     0.0195    -41.704     0.0000     -0.8496     -0.7733
# ==============================================================================

# Endogenous: prices
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','tylenol','advil','bayer']
X2=dfm[exog_vars]
model8=IV2SLS(dfm.y, X2, dfm.prices, dfm.avg_h_price).fit()
print(model8)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9013
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9013
# No. Observations:               38544   F-statistic:                 4.696e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:05:02   Distribution:                  chi2(5)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8849     0.0049    -180.79     0.0000     -0.8945     -0.8753
# prom          -0.3267     0.0189    -17.285     0.0000     -0.3638     -0.2897
# tylenol        1.7055     0.0267     63.828     0.0000      1.6531      1.7578
# advil          0.8268     0.0270     30.632     0.0000      0.7739      0.8797
# bayer         -0.7242     0.0197    -36.779     0.0000     -0.7628     -0.6856
# ==============================================================================

# Endogenous: prices
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False
exog_vars=['prom','inter1','inter2','inter3']
X3=dfm[exog_vars]
model9=IV2SLS(dfm.y, X3, dfm.prices, dfm.avg_h_price).fit()
print(model9)
#                           IV-2SLS Estimation Summary                          
# ==============================================================================
# Dep. Variable:                      y   R-squared:                      0.9046
# Estimator:                    IV-2SLS   Adj. R-squared:                 0.9041
# No. Observations:               38544   F-statistic:                 5.311e+05
# Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# Time:                        20:05:40   Distribution:                chi2(221)
# Cov. Estimator:                robust                                         
                                                                              
#                              Parameter Estimates                              
# ==============================================================================
#             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# ------------------------------------------------------------------------------
# prices        -0.8846     0.0048    -182.51     0.0000     -0.8941     -0.8751
# prom          -0.3265     0.0186    -17.573     0.0000     -0.3629     -0.2901
# (omitted)
# ==============================================================================

# Endogenous: prices
# Instruments: avg_h_price
# Robust Covariance (Heteroskedastic)
# Debiased: False



### PART THREE 
# set up pyblp
import pyblp
pyblp.options.digits=2
pyblp.options.verbose=False


# Load product data
# partition X variables into X1 and X2
# X1 is linear in parameters (X: price, prom, brands + d_j: product FEs)
# X2 is non-linear in X (the constant in X2 allows individual heterogeneity)
product_data=dfm
#note: product FEs cause collinearity with brand dummies, deleted ", absorb='C(product)')" from X1_formulation
X1_formulation=pyblp.Formulation('0 + prices + prom + tylenol + advil + bayer') 
X2_formulation=pyblp.Formulation('1 + prices + prom + tylenol + advil + bayer')
product_formulations=(X1_formulation, X2_formulation)


# Load demographic data 
agent_data=df4
agent_formulation = pyblp.Formulation('0 + income')


# specify monte carlo integration (size = 20 since we have 20 agents for each market)
mc_integration=pyblp.Integration('monte_carlo',size=20, specification_options={'seed':0})
mc_problem=pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data, integration=mc_integration)
# # mc_problem
# Dimensions:
# =========================================
#  T      N      I     K1    K2    D    MD 
# ----  -----  -----  ----  ----  ---  ----
# 3504  38544  70080   5     6     1    35 
# =========================================

# Formulations:
# =============================================================================
#        Column Indices:           0       1        2        3       4      5  
# -----------------------------  ------  ------  -------  -------  -----  -----
#  X1: Linear Characteristics    prices   prom   tylenol   advil   bayer       
# X2: Nonlinear Characteristics    1     prices   prom    tylenol  advil  bayer
#        d: Demographics         income                                        
# =============================================================================

# Optimization routines (many options in pyblp)
bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-4})
tighter_bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-5})

# Specify correlation and interaction structure across parameters
# recall from class that sigma is a square matrix of (k+1) by (k+1)
# and pi is a (k+1) by d matrix and we want to interact price with income
initial_sigma=np.diag([1,1,1,1,1,1]) 	#no correlation across X2
initial_pi=np.array([
	[0],
	[1],
	[0],
	[0],
	[0],
	[0]
	])
results = mc_problem.solve(initial_sigma,initial_pi,optimization=bfgs,method='1s')
# Problem Results Summary:
# =======================================================================================================
# GMM   Objective  Gradient      Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
# Step    Value      Norm    Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
# ----  ---------  --------  --------------  --------------  -------  ----------------  -----------------
#  1    +2.4E+03   +3.3E+03     -3.4E+03        +3.8E+03        0         +7.5E+04          +1.9E+08     
# =======================================================================================================

# Cumulative Statistics:
# ===========================================================================
# Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
#    Time      Converged   Iterations   Evaluations  Iterations   Evaluations
# -----------  ---------  ------------  -----------  -----------  -----------
#  00:51:53       No           1            17        14401902     43255680  
# ===========================================================================

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# =======================================================================================================
# Sigma:       1         prices       prom      tylenol      advil       bayer     |    Pi:      income  
# -------  ----------  ----------  ----------  ----------  ----------  ----------  |  -------  ----------
#    1      +1.0E+00                                                               |     1      +0.0E+00 
#          (+6.0E-01)                                                              |                     
#                                                                                  |                     
# prices    +0.0E+00    +6.9E-01                                                   |  prices    +4.0E-02 
#                      (+1.0E-01)                                                  |           (+5.8E-02)
#                                                                                  |                     
#  prom     +0.0E+00    +0.0E+00    +1.0E+00                                       |   prom     +0.0E+00 
#                                  (+1.6E+00)                                      |                     
#                                                                                  |                     
# tylenol   +0.0E+00    +0.0E+00    +0.0E+00    +9.8E-01                           |  tylenol   +0.0E+00 
#                                              (+1.1E+00)                          |                     
#                                                                                  |                     
#  advil    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +1.0E+00               |   advil    +0.0E+00 
#                                                          (+1.7E+00)              |                     
#                                                                                  |                     
#  bayer    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +1.0E+00   |   bayer    +0.0E+00 
#                                                                      (+8.6E-01)  |                     
# =======================================================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ==========================================================
#   prices       prom      tylenol      advil       bayer   
# ----------  ----------  ----------  ----------  ----------
#  -1.7E+00    -1.3E-01    +1.8E+00    +9.0E-01    -5.7E-01 
# (+6.3E-01)  (+7.2E-01)  (+3.6E-01)  (+8.3E-01)  (+5.8E-01)
# ==========================================================
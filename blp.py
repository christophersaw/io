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


# calculate market ids and shares
df['market']=df.groupby(['store','week']).ngroup()
df['x']='x'
df['market_ids']=df['store'].astype(str)+df['x'].astype(str)+df['week'].astype(str)
df['shares']=df['sales']/df['count']
del(df['x'])


# calculate inside and outside shares
df['insideshare']=df.groupby(['market'])['shares'].transform('sum')
# check that insideshares are between 0 and 1
# df['insideshare'].min()
# df['insideshare'].max()
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



# ### PART TWO: OLS/IV Logit regressions
# # Create dependent variable: Y = log(s_jt) - log(s_0t)
# dfm['log_share']=dfm['shares'].apply(lambda x: np.log(x))
# dfm['log_outside']=dfm['outsideshare'].apply(lambda x: np.log(x))
# dfm['y']=dfm['log_share']-dfm['log_outside']

# import linearmodels
# from linearmodels import OLS
# from linearmodels.iv import IV2SLS
# from linearmodels.iv.absorbing import AbsorbingLS

# exog_vars=['prices','prom']
# X1=dfm[exog_vars]
# model1 = OLS(dfm.y, X1).fit()
# #print(model1)
# #                             OLS Estimation Summary                            
# # ==============================================================================
# # Dep. Variable:                      y   R-squared:                      0.8823
# # Estimator:                        OLS   Adj. R-squared:                 0.8823
# # No. Observations:               38544   F-statistic:                 4.091e+05
# # Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# # Time:                        09:33:40   Distribution:                  chi2(2)
# # Cov. Estimator:                robust                                         
                                                                              
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prices        -1.6235     0.0028    -578.07     0.0000     -1.6290     -1.6180
# # prom          -2.5092     0.0346    -72.435     0.0000     -2.5771     -2.4413
# # ==============================================================================

# exog_vars=['prices','prom','tylenol','advil','bayer']
# X2=dfm[exog_vars]
# model2 = OLS(dfm.y, X2).fit()
# #print(model2)
# #                             OLS Estimation Summary                            
# # ==============================================================================
# # Dep. Variable:                      y   R-squared:                      0.9082
# # Estimator:                        OLS   Adj. R-squared:                 0.9082
# # No. Observations:               38544   F-statistic:                 7.133e+05
# # Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# # Time:                        09:33:40   Distribution:                  chi2(5)
# # Cov. Estimator:                robust                                         
                                                                              
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prices        -1.6453     0.0112    -146.77     0.0000     -1.6672     -1.6233
# # prom          -1.6077     0.0369    -43.602     0.0000     -1.6800     -1.5354
# # tylenol        1.2412     0.0600     20.701     0.0000      1.1237      1.3587
# # advil          0.3469     0.0593     5.8540     0.0000      0.2307      0.4630
# # bayer         -2.1471     0.0434    -49.480     0.0000     -2.2321     -2.0620
# # ==============================================================================

# brandstoredummies = pd.DataFrame({'inter1': pd.Categorical(dfm.inter1), 'inter2': pd.Categorical(dfm.inter2),'inter3': pd.Categorical(dfm.inter3)})
# model3=AbsorbingLS(dfm.y, X2, absorb=brandstoredummies, drop_absorbed=True).fit()
# #print(model3)
# #                          Absorbing LS Estimation Summary                          
# # ==================================================================================
# # Dep. Variable:                      y   R-squared:                          0.4858
# # Estimator:               Absorbing LS   Adj. R-squared:                     0.4828
# # No. Observations:               38544   F-statistic:                      2.21e+04
# # Date:                Wed, Oct 27 2021   P-value (F-stat):                   0.0000
# # Time:                        09:29:18   Distribution:                      chi2(2)
# # Cov. Estimator:                robust   R-squared (No Effects):             0.3064
# #                                         Variables Absorbed:                 220.00
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prices        -0.3870     0.0028    -138.44     0.0000     -0.3924     -0.3815
# # prom           0.3613     0.0144     25.037     0.0000      0.3330      0.3896
# # ==============================================================================

# exog_vars=['prom']
# X1=dfm[exog_vars]
# model4=IV2SLS(dfm.y, X1, dfm.prices, dfm.demand_instruments0).fit()
# #print(model4)
# #                           IV-2SLS Estimation Summary                          
# # ==============================================================================
# # Dep. Variable:                      y   R-squared:                      0.8823
# # Estimator:                    IV-2SLS   Adj. R-squared:                 0.8823
# # No. Observations:               38544   F-statistic:                 4.511e+05
# # Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# # Time:                        10:14:26   Distribution:                  chi2(2)
# # Cov. Estimator:                robust                                         
                                                                              
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prom          -2.5362     0.0344    -73.686     0.0000     -2.6036     -2.4687
# # prices        -1.6172     0.0027    -609.93     0.0000     -1.6224     -1.6120
# # ==============================================================================

# # Endogenous: prices
# # Instruments: demand_instruments0
# # Robust Covariance (Heteroskedastic)
# # Debiased: False

# exog_vars=['prom','tylenol','advil','bayer']
# X2=dfm[exog_vars]
# model5=IV2SLS(dfm.y, X2, dfm.prices, dfm.demand_instruments0).fit()
# # print(model5)
# #                           IV-2SLS Estimation Summary                          
# # ==============================================================================
# # Dep. Variable:                      y   R-squared:                      0.9079
# # Estimator:                    IV-2SLS   Adj. R-squared:                 0.9079
# # No. Observations:               38544   F-statistic:                 7.964e+05
# # Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# # Time:                        10:16:17   Distribution:                  chi2(5)
# # Cov. Estimator:                robust                                         
                                                                              
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prom          -1.6106     0.0372    -43.283     0.0000     -1.6836     -1.5377
# # tylenol        0.7570     0.0601     12.604     0.0000      0.6393      0.8747
# # advil         -0.1329     0.0594    -2.2385     0.0252     -0.2492     -0.0165
# # bayer         -2.4930     0.0430    -57.994     0.0000     -2.5772     -2.4087
# # prices        -1.5502     0.0112    -138.30     0.0000     -1.5722     -1.5283
# # ==============================================================================

# # Endogenous: prices
# # Instruments: demand_instruments0
# # Robust Covariance (Heteroskedastic)
# # Debiased: False

# dfm['inter1']=dfm['inter1'].astype('category')
# dfm['inter2']=dfm['inter2'].astype('category')
# dfm['inter3']=dfm['inter3'].astype('category')
# # set store 9 in week 10 as the base
# dfm.loc[dfm['market_ids']=="9x10", 'inter1'] = 0
# dfm.loc[dfm['market_ids']=="9x10", 'inter2'] = 0
# dfm.loc[dfm['market_ids']=="9x10", 'inter3'] = 0 

# exog_vars=['prom','inter1','inter2','inter3']
# X3=dfm[exog_vars]
# model6=IV2SLS(dfm.y, X3, dfm.prices, dfm.demand_instruments0).fit()
# # print(model6)

# #                           IV-2SLS Estimation Summary                          
# # ==============================================================================
# # Dep. Variable:                      y   R-squared:                      0.9086
# # Estimator:                    IV-2SLS   Adj. R-squared:                 0.9081
# # No. Observations:               38544   F-statistic:                 8.435e+05
# # Date:                Wed, Oct 27 2021   P-value (F-stat)                0.0000
# # Time:                        11:13:42   Distribution:                chi2(221)
# # Cov. Estimator:                robust                                         
                                                                              
# #                              Parameter Estimates                              
# # ==============================================================================
# #             Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
# # ------------------------------------------------------------------------------
# # prom          -1.6143     0.0370    -43.596     0.0000     -1.6868     -1.5417
# # (omitted)
# # prices        -1.5522     0.0111    -139.33     0.0000     -1.5740     -1.5304
# # ==============================================================================

# # Endogenous: prices
# # Instruments: demand_instruments0
# # Robust Covariance (Heteroskedastic)
# # Debiased: False



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
initial_sigma=np.diag([0.001,0.001,0.001,0.001,0.001,0.001]) 	#no correlation across X2
initial_pi=np.array([
	[0],
	[0.001],
	[0],
	[0],
	[0],
	[0]
	])
results = mc_problem.solve(initial_sigma,initial_pi,optimization=bfgs,method='1s')
#results
                                                                                                                                                                                            
# Problem Results Summary:
# =======================================================================================================
# GMM   Objective  Gradient      Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
# Step    Value      Norm    Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
# ----  ---------  --------  --------------  --------------  -------  ----------------  -----------------
#  1    +1.3E+03   +8.7E-06     +3.8E+00        +1.2E+04        0         +1.1E+04          +9.3E+08     
# =======================================================================================================

# Cumulative Statistics:
# ===========================================================================
# Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
#    Time      Converged   Iterations   Evaluations  Iterations   Evaluations
# -----------  ---------  ------------  -----------  -----------  -----------
#  01:25:09       Yes          33           41        20756676     62415753  
# ===========================================================================

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# =======================================================================================================
# Sigma:       1         prices       prom      tylenol      advil       bayer     |    Pi:      income  
# -------  ----------  ----------  ----------  ----------  ----------  ----------  |  -------  ----------
#    1      +4.6E-01                                                               |     1      +0.0E+00 
#          (+1.6E+00)                                                              |                     
#                                                                                  |                     
# prices    +0.0E+00    +1.0E+00                                                   |  prices    +5.0E-01 
#                      (+2.3E-01)                                                  |           (+1.7E-01)
#                                                                                  |                     
#  prom     +0.0E+00    +0.0E+00    -1.5E-01                                       |   prom     +0.0E+00 
#                                  (+3.6E+00)                                      |                     
#                                                                                  |                     
# tylenol   +0.0E+00    +0.0E+00    +0.0E+00    +1.3E+01                           |  tylenol   +0.0E+00 
#                                              (+5.1E+00)                          |                     
#                                                                                  |                     
#  advil    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +1.1E+01               |   advil    +0.0E+00 
#                                                          (+7.6E+00)              |                     
#                                                                                  |                     
#  bayer    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +1.0E+01   |   bayer    +0.0E+00 
#                                                                      (+3.5E+00)  |                     
# =======================================================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ==========================================================
#   prices       prom      tylenol      advil       bayer   
# ----------  ----------  ----------  ----------  ----------
#  -8.3E+00    -2.1E+00    -1.5E+01    -1.2E+01    -1.5E+01 
# (+1.8E+00)  (+2.8E-01)  (+9.2E+00)  (+1.2E+01)  (+5.9E+00)
# ==========================================================

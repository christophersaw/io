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
df.loc[dfm['product'].isin([1,2,3]), 'tylenol']=1
df['advil']=0
df.loc[dfm['product'].isin([4,5,6]), 'advil']=1
df['bayer']=0
df.loc[dfm['product'].isin([7,8,9]), 'bayer']=1


# calculate market ids and shares
df['market_ids']=df.groupby(['store','week']).ngroup()
df['shares']=df['sales']/df['count']


# calculate inside and outside shares
df['insideshare']=df.groupby(['market_ids'])['shares'].transform('sum')
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
df3['market_ids']=df3.groupby(['store','week']).ngroup()
df4=pd.melt(df3, id_vars=['market_ids'], value_vars=[
	'hhincome1','hhincome2','hhincome3','hhincome4','hhincome5','hhincome6','hhincome7','hhincome8','hhincome9','hhincome10',
	'hhincome11','hhincome12','hhincome13','hhincome14','hhincome15','hhincome16','hhincome17','hhincome18','hhincome19','hhincome20'], 
	var_name='agent_index', value_name='income')
del(df4['agent_index'])
#dfprint=df4.sort_values(by=['market_ids'])
#dfprint.to_csv('agents_sorted.csv',index=False)



### PART TWO: OLS/IV Logit regressions
# Create dependent variable: Y = log(s_jt) - log(s_0t)
dfm['log_share']=dfm['shares'].apply(lambda x: np.log(x))
dfm['log_outside']=dfm['outsideshare'].apply(lambda x: np.log(x))
dfm['y']=dfm['log_share']-dfm['log_outside']


import stata_setup
stata_setup.config("/Applications/Stata/", "se")
from pystata import stata
stata.pdataframe_to_data(dfm, force=True)
stata.run('''
cd "/Users/christophersaw/Desktop/blp"
local model_1 prom
local model_2 prom tylenol advil bayer
local model_3 prom tylenol advil bayer tylenol#i.store advil#i.store bayer#i.store

* Questions 1 to 3 (OLS/Logit model)
forvalues m = 1/3{
	quietly reg y prices `model_`m'', noconstant
	outreg2 using PS1Q1.csv, append
}

* Questions 4 and 5 (IV/Logit model)
* Hausman instruments
quietly {
	forvalues t = 0/3503 {
		gen dummy = 1
		replace dummy = 0 if market_ids == `t'
		gen price2 = dummy*prices
		replace price2 = . if price2==0
		bysort product week: egen price3_`t' = mean(price2)
		replace price3_`t' = 0 if market_ids != `t'
		drop dummy price2
	}
	egen hausmanprice = rowtotal(price3_*)
	drop price3_* 
}
forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (prices = cost), noconstant
	outreg2 using PS1Q1.csv, append
}

forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (prices = hausmanprice), noconstant
	outreg2 using PS1Q1.csv, append
}
	''')

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

# Optimization routine (many options in pyblp)
bfgs=pyblp.Optimization('bfgs', {'gtol': 1e-4})

# see https://pyblp.readthedocs.io/en/stable/background.html#equation-w for the weighting matrix

# Specify correlation and interaction structure across parameters
# recall from class that sigma is a square matrix of (k+1) by (k+1)
# and pi is a (k+1) by d matrix and we want to interact price with income
initial_sigma=np.diag([1,1,1,1,1,1]) 	#no correlation across X2; for full correlation we put np.ones((6,6))
initial_pi=np.array([
[0],
[1],
[0],
[0],
[0],
[0]
	])
tighter_bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-5})
results = mc_problem.solve(initial_sigma,initial_pi,optimization=tighter_bfgs,method='1s')
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


# allow correlation across prom and brands
initial_sigma2=np.array([
[1,0,0,0,0,0],
[0,1,0,0,0,0],
[0,0,1,0,0,0],
[0,0,1,1,0,0],
[0,0,1,1,1,0],
[0,0,1,1,1,1],
]) 	
results2 = mc_problem.solve(initial_sigma2,initial_pi,optimization=tighter_bfgs,method='1s')
#results2                                                                                                                                                                                             
# Problem Results Summary:
# =======================================================================================================
# GMM   Objective  Gradient      Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
# Step    Value      Norm    Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
# ----  ---------  --------  --------------  --------------  -------  ----------------  -----------------
#  1    +3.8E+02   +8.1E-06     +4.3E+00        +3.2E+03        0         +1.1E+04          +5.2E+08     
# =======================================================================================================

# Cumulative Statistics:
# ===========================================================================
# Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
#    Time      Converged   Iterations   Evaluations  Iterations   Evaluations
# -----------  ---------  ------------  -----------  -----------  -----------
#  00:41:55       Yes          68           83        10653758     32213280  
# ===========================================================================

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# ==================================================================================================================================================================================================
# Sigma:       1         prices       prom      tylenol      advil       bayer     |  Sigma Squared:      1         prices       prom      tylenol      advil       bayer     |    Pi:      income  
# -------  ----------  ----------  ----------  ----------  ----------  ----------  |  --------------  ----------  ----------  ----------  ----------  ----------  ----------  |  -------  ----------
#    1      +2.7E-01                                                               |        1          +7.5E-02    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00   |     1      +0.0E+00 
#          (+1.8E+00)                                                              |                  (+9.9E-01)  (+0.0E+00)  (+0.0E+00)  (+0.0E+00)  (+0.0E+00)  (+0.0E+00)  |                     
#                                                                                  |                                                                                          |                     
# prices    +0.0E+00    +1.4E+00                                                   |      prices       +0.0E+00    +2.0E+00    +0.0E+00    +0.0E+00    +0.0E+00    +0.0E+00   |  prices    +3.2E-01 
#                      (+3.7E-01)                                                  |                  (+0.0E+00)  (+1.1E+00)  (+0.0E+00)  (+0.0E+00)  (+0.0E+00)  (+0.0E+00)  |           (+2.7E-01)
#                                                                                  |                                                                                          |                     
#  prom     +0.0E+00    +0.0E+00    +3.4E-01                                       |       prom        +0.0E+00    +0.0E+00    +1.1E-01    +2.6E+00    +1.4E+00    -2.0E+00   |   prom     +0.0E+00 
#                                  (+3.0E+00)                                      |                  (+0.0E+00)  (+0.0E+00)  (+2.0E+00)  (+2.3E+01)  (+1.2E+01)  (+1.8E+01)  |                     
#                                                                                  |                                                                                          |                     
# tylenol   +0.0E+00    +0.0E+00    +7.8E+00    -5.9E+00                           |     tylenol       +0.0E+00    +0.0E+00    +2.6E+00    +9.5E+01    +6.5E+01    -8.4E+01   |  tylenol   +0.0E+00 
#                                  (+3.4E+00)  (+3.9E+00)                          |                  (+0.0E+00)  (+0.0E+00)  (+2.3E+01)  (+6.4E+01)  (+7.5E+01)  (+3.8E+01)  |                     
#                                                                                  |                                                                                          |                     
#  advil    +0.0E+00    +0.0E+00    +4.1E+00    -5.6E+00    +1.3E+00               |      advil        +0.0E+00    +0.0E+00    +1.4E+00    +6.5E+01    +5.0E+01    -6.5E+01   |   advil    +0.0E+00 
#                                  (+9.1E+00)  (+5.0E+00)  (+6.4E+00)              |                  (+0.0E+00)  (+0.0E+00)  (+1.2E+01)  (+7.5E+01)  (+8.2E+01)  (+5.7E+01)  |                     
#                                                                                  |                                                                                          |                     
#  bayer    +0.0E+00    +0.0E+00    -5.9E+00    +6.4E+00    -3.6E+00    +5.8E+00   |      bayer        +0.0E+00    +0.0E+00    -2.0E+00    -8.4E+01    -6.5E+01    +1.2E+02   |   bayer    +0.0E+00 
#                                  (+4.8E+00)  (+3.4E+00)  (+2.4E+00)  (+3.1E+00)  |                  (+0.0E+00)  (+0.0E+00)  (+1.8E+01)  (+3.8E+01)  (+5.7E+01)  (+1.1E+02)  |                     
# ==================================================================================================================================================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ==========================================================
#   prices       prom      tylenol      advil       bayer   
# ----------  ----------  ----------  ----------  ----------
#  -7.1E+00    -2.1E+00    -8.7E+00    -5.9E+00    -1.5E+01 
# (+3.2E+00)  (+2.2E-01)  (+5.7E+00)  (+7.6E+00)  (+7.6E+00)
# ==========================================================
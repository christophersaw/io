import os
import pandas as pd
import numpy as np
import pyblp
os.chdir("/Users/christophersaw/Desktop/blp")
data=pd.read_csv(r'headache.csv')

# Rename columns for pyblp syntax
data=data.rename(columns={'price': 'prices'})
data=data.rename(columns={'cost': 'demand_instruments0',
'pricestore1': 'demand_instruments1','pricestore2': 'demand_instruments2','pricestore3': 'demand_instruments3','pricestore4': 'demand_instruments4',
'pricestore5': 'demand_instruments5','pricestore6': 'demand_instruments6','pricestore7': 'demand_instruments7','pricestore8': 'demand_instruments8',
'pricestore9': 'demand_instruments9','pricestore10': 'demand_instruments10','pricestore11': 'demand_instruments11','pricestore12': 'demand_instruments12',
'pricestore13': 'demand_instruments13','pricestore14': 'demand_instruments14','pricestore15': 'demand_instruments15','pricestore16': 'demand_instruments16',
'pricestore17': 'demand_instruments17','pricestore18': 'demand_instruments18','pricestore19': 'demand_instruments19','pricestore20': 'demand_instruments20',
'pricestore21': 'demand_instruments21','pricestore22': 'demand_instruments22','pricestore23': 'demand_instruments23','pricestore24': 'demand_instruments24',
'pricestore25': 'demand_instruments25','pricestore26': 'demand_instruments26','pricestore27': 'demand_instruments27','pricestore28': 'demand_instruments28',
'pricestore29': 'demand_instruments29','pricestore30': 'demand_instruments30', 'avoutprice': 'demand_instruments31'})

# Load demographic data and reshape for pyblp (each row is an agent in market t)
incdata=pd.read_csv(r'OTCDemographics.csv',sep='\t')
incdata['market_ids']=incdata['store'].astype(str)+str('x')+incdata['week'].astype(str)
incdata=pd.melt(incdata, id_vars=['market_ids'], value_vars=[
	'hhincome1','hhincome2','hhincome3','hhincome4','hhincome5','hhincome6','hhincome7','hhincome8','hhincome9','hhincome10',
	'hhincome11','hhincome12','hhincome13','hhincome14','hhincome15','hhincome16','hhincome17','hhincome18','hhincome19','hhincome20'], 
	var_name='agent_index', value_name='income')

# Configure pyblp
pyblp.options.digits=2
pyblp.options.verbose=False
product_data=data
X1_formulation=pyblp.Formulation('0 + prices + prom + C(brand)') 
X2_formulation=pyblp.Formulation('0 + prices + branded_product')
product_formulations=(X1_formulation, X2_formulation)
agent_data=incdata
agent_formulation = pyblp.Formulation('0 + income')
mc_integration=pyblp.Integration('monte_carlo',size=20, specification_options={'seed':0})
mc_problem=pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data, integration=mc_integration)
# =========================================
#  T      N      I     K1    K2    D    MD 
# ----  -----  -----  ----  ----  ---  ----
# 3504  38544  70080   13    2     1    44 
# =========================================

# Formulations:
# ======================================================================================================================================================================
#        Column Indices:           0            1            2         3         4         5         6         7         8         9         10        11         12    
# -----------------------------  ------  ---------------  --------  --------  --------  --------  --------  --------  --------  --------  --------  ---------  ---------
#  X1: Linear Characteristics    prices       prom        brand[1]  brand[2]  brand[3]  brand[4]  brand[5]  brand[6]  brand[7]  brand[8]  brand[9]  brand[10]  brand[11]
# X2: Nonlinear Characteristics  prices  branded_product                                                                                                                
#        d: Demographics         income                                                                                                                                 
# ======================================================================================================================================================================

# Load optimization routine
neldermead=pyblp.Optimization('nelder-mead',compute_gradient=False) 

# Specify correlation and interaction structure across parameters
initial_sigma=np.diag([0,0.001])
initial_pi=np.array([
	[0.001],
	[0]
	])
results = mc_problem.solve(initial_sigma,initial_pi,optimization=neldermead)
# Problem Results Summary:
# =======================================================================================================
# GMM   Objective  Gradient      Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
# Step    Value      Norm    Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
# ----  ---------  --------  --------------  --------------  -------  ----------------  -----------------
#  2    +6.9E+02   +1.7E-03     +3.9E+01        +2.3E+03        0         +8.4E+07          +5.9E+07     
# =======================================================================================================

# Cumulative Statistics:
# ===========================================================================
# Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
#    Time      Converged   Iterations   Evaluations  Iterations   Evaluations
# -----------  ---------  ------------  -----------  -----------  -----------
#  00:09:53       Yes          94           187        1509760      4705046  
# ===========================================================================

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# ==========================================================================
#     Sigma:        prices   branded_product  |        Pi:          income  
# ---------------  --------  ---------------  |  ---------------  ----------
#     prices       +0.0E+00                   |      prices        +1.4E-01 
#                                             |                   (+2.5E-02)
#                                             |                             
# branded_product  +0.0E+00     -2.1E-01      |  branded_product   +0.0E+00 
#                              (+3.4E-01)     |                             
# ==========================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ==========================================================================================================================================================
#   prices       prom      brand[1]    brand[2]    brand[3]    brand[4]    brand[5]    brand[6]    brand[7]    brand[8]    brand[9]   brand[10]   brand[11] 
# ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------
#  -1.7E+00    +3.9E-01    -6.5E+00    -6.0E+00    -6.0E+00    -6.9E+00    -6.9E+00    -7.2E+00    -8.2E+00    -8.1E+00    -6.8E+00    -7.5E+00    -7.1E+00 
# (+3.0E-01)  (+1.4E-02)  (+1.4E-01)  (+1.3E-01)  (+1.1E-01)  (+1.3E-01)  (+1.3E-01)  (+1.2E-01)  (+1.3E-01)  (+1.1E-01)  (+9.0E-02)  (+5.4E-02)  (+6.0E-02)
# ==========================================================================================================================================================



### bfgs=pyblp.Optimization('bfgs',{'gtol': 1e-5})
### results = mc_problem.solve(initial_sigma,initial_pi,optimization=bfgs,method='1s')
# Problem Results Summary:
# =======================================================================================================
# GMM   Objective  Gradient      Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
# Step    Value      Norm    Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
# ----  ---------  --------  --------------  --------------  -------  ----------------  -----------------
#  1    +3.5E+02   +4.5E-06     -7.1E+00        +1.2E+03        0         +8.9E+07          +4.3E+07     
# =======================================================================================================

# Cumulative Statistics:
# ===========================================================================
# Computation  Optimizer  Optimization   Objective   Fixed Point  Contraction
#    Time      Converged   Iterations   Evaluations  Iterations   Evaluations
# -----------  ---------  ------------  -----------  -----------  -----------
#  00:04:07       Yes          8            15         1409621      4262566  
# ===========================================================================

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# ==========================================================================
#     Sigma:        prices   branded_product  |        Pi:          income  
# ---------------  --------  ---------------  |  ---------------  ----------
#     prices       +0.0E+00                   |      prices        +1.6E-01 
#                                             |                   (+2.5E-02)
#                                             |                             
# branded_product  +0.0E+00     -1.0E-01      |  branded_product   +0.0E+00 
#                              (+3.3E-01)     |                             
# ==========================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ==========================================================================================================================================================
#   prices       prom      brand[1]    brand[2]    brand[3]    brand[4]    brand[5]    brand[6]    brand[7]    brand[8]    brand[9]   brand[10]   brand[11] 
# ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------
#  -2.0E+00    +3.8E-01    -6.4E+00    -5.9E+00    -5.9E+00    -6.8E+00    -6.8E+00    -7.1E+00    -8.1E+00    -8.0E+00    -6.7E+00    -7.4E+00    -7.1E+00 
# (+3.0E-01)  (+1.4E-02)  (+1.2E-01)  (+1.1E-01)  (+9.3E-02)  (+1.2E-01)  (+1.1E-01)  (+1.0E-01)  (+1.1E-01)  (+9.4E-02)  (+6.7E-02)  (+5.9E-02)  (+6.6E-02)
# ==========================================================================================================================================================
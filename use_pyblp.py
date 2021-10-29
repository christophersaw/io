import os
import pandas as pd
import numpy as np
import pyblp
os.chdir("/Users/christophersaw/Desktop/blp")
df=pd.read_csv(r'headache.csv')
df2=pd.read_csv(r'headache_instr.csv')

# Rename columns (prices and demand_instruments) for pyblp syntax, merge dataframes
df=df.rename(columns={'price': 'prices'})
list1=['store','week','brand','demand_instruments0','demand_instruments1']
list2=['demand_instruments'+str(i) for i in range(2,32)]
del(df2['weight'])
col_list=list1+list2
df2.columns=col_list
data=pd.merge(df, df2, on=['store','week','brand']) # this is a one-one merge

# Load demographic data and reshape for pyblp (each row is an agent in market t)
incdata=pd.read_csv(r'OTCDemographics.csv',sep='\t')
incdata['market_ids']=incdata['store'].astype(str)+str('x')+incdata['week'].astype(str)
incdata=pd.melt(incdata, id_vars=['market_ids'],value_vars=['hhincome'+str(i) for i in range(1,21)],var_name='agent_index',value_name='income')
del(incdata['agent_index'])

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
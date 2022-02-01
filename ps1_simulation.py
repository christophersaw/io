import pandas as pd
import numpy as np

# (1a) Set seed and parameters
np.random.seed(123456789)
N=1000
T=50
beta_zero=0
beta_k=0.4
beta_l=0.6
rho=0.7
theta=0.3
delta=0.2
r=0.05

# (1b) Initial conditions for productivity and wages
sigma_omega=np.sqrt(0.3)
sigma_logwage=np.sqrt(0.1)
omega_zero=np.random.normal(0,sigma_omega,(N,1)) # omega_0 ~ N(0,0.3)
wage_zero=np.exp(np.random.normal(0,sigma_logwage,(N,1))) # log wage_0 ~ N(0,0.1)

# (1b) Shocks to productivity and wages (i.i.d. across firms and time)
sigma_xi=np.sqrt(0.214)
sigma_nu=np.sqrt(0.1)
xi=np.random.normal(0,sigma_xi,(N,T)) # xi ~ N(0,0.214)
nu=np.exp(np.random.normal(0,sigma_nu,(N,T))) # log nu ~ N(0,0.1) 

# (1c) Paths for productivity and wages
M=np.zeros((N,T))
omega=np.concatenate((omega_zero,M),axis=1)	
wage=np.concatenate((wage_zero,M),axis=1)	

for t in range(T):
	omega[:,t+1]=rho*omega[:,t]+xi[:,t] # Productivity path from t=0 to t=50			
	wage[:,t+1]=theta*wage[:,t]+nu[:,t] # Wage path from t=0 to t=50


# (1c) Generate time-invariant investment costs for firms
sigma_g=np.sqrt(0.6)
g=np.random.normal(0,sigma_g,(N,1)) # g = log(1/gamma)~N(0,0.6)
gamma=1/np.exp(g) # gamma = 1/exp(g)

# (2d) Path for investments from t=0 to t=49 (use FOC for investments)
inv=np.zeros((N,T))
for i in range(N):
	for t in range(T):
		inv[i,t]=1/gamma[i]*beta_k*np.exp(omega[i,t+1])*\
		np.power((np.exp(omega[i,t+1])*beta_l/wage[i,t+1]),(beta_l/(1-beta_l)))


# (2d) Path for capital from t=0 to t=50 (use equation for capital growth)
# Assume that every firm begins with 1 unit of capital at t=0
capital=np.ones((N,T+1)) 									
for t in range(T):
	capital[:,t+1]=(1-delta)*capital[:,t]+inv[:,t]


log_capital=np.log(capital)

# (2e) Path for labor from t=0 to t=50 (use FOC for labor)
labor=np.power((np.exp(omega)*np.power(capital,beta_k)*beta_l)/wage,1/(1-beta_l))
log_labor=np.log(labor)

# (2f) Simulate measurement error
sigma_err=np.sqrt(0.1)
err=np.random.normal(0,sigma_err,(N,T+1))					

# (2f) Path for output (use equation 1)
log_output=beta_zero+beta_k*log_capital+beta_l*log_labor+omega+err
output=np.exp(log_output)

# (2g) Path for materials (output prior to measurement error)
log_materials=log_output-omega-err
materials=np.exp(log_materials)

# (3) Dataframe for q, k, l, m
# Index for firm_ids and time periods
firms=pd.DataFrame(np.linspace(1,N,num=N), columns=["firm_id"])
firm_id=pd.concat([firms]*51,ignore_index=True)
time=pd.Series(np.linspace(0,T,num=T+1)).repeat(1000)
time=pd.DataFrame(time, columns=["time"]).reset_index()
# Reshape arrays and assemble dataframe
q_it=pd.DataFrame(log_output.flatten('F'), columns=["log_output"])
k_it=pd.DataFrame(log_capital.flatten('F'), columns=["log_capital"])
l_it=pd.DataFrame(log_labor.flatten('F'), columns=["log_labor"])
m_it=pd.DataFrame(log_materials.flatten('F'), columns=["log_materials"])

# (3) Assemble dataframe
dataframe=firm_id.join(time['time']).join(q_it).join(k_it).join(l_it).join(m_it)
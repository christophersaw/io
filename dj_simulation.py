import os
os.chdir("/Users/christophersaw/Desktop/Replication")
import pandas as pd
import numpy as np

# Setup
np.random.seed(123456789)
N=300
T=10
k=np.zeros((N,T+1)) # log capital
l=np.zeros((N,T+1)) # log labor
m=np.zeros((N,T+1)) # log materials
k1=np.zeros((N,T+1)) # 1 denotes lag
l1=np.zeros((N,T+1))
m1=np.zeros((N,T+1))	

# Parameters
nu=0.933
sigma=0.695
eta=2.431
beta_k=0.137
beta_m=1-beta_k

# Capital, Labor, Materials
# Assume that the initial distribution of firms is lognormal(0,1) and k0=m0=l0
# From the initial distribution, growth of (k, l, m) is stochastic 
# Follows Industry 3 in D&J (Table 1)
S=np.random.normal(0,1,(N,1))
for i in range(N):
	k[i,0]=S[i]
	l[i,0]=S[i]
	m[i,0]=S[i]
	for t in range(T):
		k[i,t+1]=k[i,t]+np.random.normal(0.062,0.182)
		l[i,t+1]=l[i,t]+np.random.normal(0.015,0.170)
		m[i,t+1]=m[i,t]+np.random.normal(0.044,0.274)
		k1[i,t+1]=k[i,t]
		l1[i,t+1]=l[i,t]
		m1[i,t+1]=m[i,t]

# Productivity
z=np.zeros((N,T+1)) # log Z (such as R&D expenditures)
WL=np.zeros((N,T+1)) # labor-augmenting productivity
WH=np.zeros((N,T+1)) # Hicks-neutral productivity
z1=np.zeros((N,T+1))		
	
# Assume that z0 is proportional to the initial state
# Assume that z follows a MA(1) process
# Assume that WL, WH are randomly drawn in the initial state
e0=np.random.normal(0,1,(N,T+1))
e1=np.random.normal(0,1,(N,T+1))
xi1=np.random.normal(0,1,(N,T+1))
xi2=np.random.normal(0,1,(N,T+1))
alpha=0.5
rho=0.8
theta=0.2
for i in range(N):
	z[i,0]=0.5*S[i]
	WL[i,0]=np.random.uniform(0,1)
	WH[i,0]=np.random.uniform(0,1)
	for t in range(T):
		z[i,t+1]=z[i,0]+alpha*e1[i,t]+e1[i,t+1]
		WL[i,t+1]=rho*WL[i,t]+theta*z[i,t]+xi1[i,t+1]
		WH[i,t+1]=rho*WH[i,t]+theta*z[i,t]+xi2[i,t+1]
		z1[i,t+1]=z[i,t]

# Output (follows CES production function)
Q=np.zeros((N,T+1)) # Q_it (levels)
X=np.zeros((N,T+1)) # X_it
p=(sigma-1)/sigma
beta_l=np.exp(WL)
for t in range(T+1):
	for i in range(N):
		X[i,t]=beta_k*np.power(np.exp(k[i,t]),p)+\
		np.power(beta_l[i,t]*np.exp(l[i,t]),p)+\
		beta_m*np.power(np.exp(m[i,t]),p)
		Q[i,t]=np.power(X[i,t],nu/p)*np.exp(WH[i,t])*np.exp(e0[i,t])

q=np.log(Q)
x=np.log(X)

# Goods price
D=np.zeros((N,T+1))	# D_it
ed=np.random.normal(0,1,(N,T+1))
for i in range(N):
	D[i,0]=np.random.uniform(0,1)
	for t in range(T):
		D[i,t+1]=D[i,0]+alpha*ed[i,t]+ed[i,t+1]

p=D-eta*q

# Wages and Price of materials
# Use FOCs in logs
C1=np.log((1+eta)*nu)
C2=np.log((1+eta)*nu*beta_m)
C3=(1+nu*sigma-sigma)/(sigma-1)
C4=-(1/sigma)

w=C1+C3*x+WL+C4*l+p+WH+e0
pm=C2+C3*x+C4*m+p+WH+e0

# Lags for output, demand shifter and prices
q1=np.zeros((N,T+1))
p1=np.zeros((N,T+1))
D1=np.zeros((N,T+1))
w1=np.zeros((N,T+1))
pm1=np.zeros((N,T+1))
for i in range(N):
	for t in range(T):
		q1[i,t+1]=q[i,t]
		p1[i,t+1]=p[i,t]
		D1[i,t+1]=D[i,t]
		w1[i,t+1]=w[i,t]
		pm1[i,t+1]=pm[i,t]

# Reshape
q=pd.DataFrame(q.flatten('F'), columns=["q"])
q1=pd.DataFrame(q1.flatten('F'), columns=["q1"])
p=pd.DataFrame(p.flatten('F'), columns=["p"])
p1=pd.DataFrame(p1.flatten('F'), columns=["p1"])
D=pd.DataFrame(D.flatten('F'), columns=["D"])
D1=pd.DataFrame(D1.flatten('F'), columns=["D1"])
k=pd.DataFrame(k.flatten('F'), columns=["k"])
k1=pd.DataFrame(k1.flatten('F'), columns=["k1"])
l=pd.DataFrame(l.flatten('F'), columns=["l"])
l1=pd.DataFrame(l1.flatten('F'), columns=["l1"])
m=pd.DataFrame(m.flatten('F'), columns=["m"])
m1=pd.DataFrame(m1.flatten('F'), columns=["m1"])
z=pd.DataFrame(z.flatten('F'), columns=["z"])
z1=pd.DataFrame(z1.flatten('F'), columns=["z1"])
w=pd.DataFrame(w.flatten('F'), columns=["w"])
w1=pd.DataFrame(w1.flatten('F'), columns=["w1"])
pm=pd.DataFrame(pm.flatten('F'), columns=["pm"])
pm1=pd.DataFrame(pm1.flatten('F'), columns=["pm1"])
WL=pd.DataFrame(WL.flatten('F'), columns=["WL"])
WH=pd.DataFrame(WH.flatten('F'), columns=["WH"])

# Firm and time ids
firm=pd.DataFrame(np.linspace(1,N,num=N), columns=["firm_id"])
firm=pd.concat([firm]*(T+1),ignore_index=True)
time=pd.Series(np.linspace(0,T,num=T+1)).repeat(N)
time=pd.DataFrame(time, columns=["time"]).reset_index()

# Build dataframe
df=firm.join(time['time']).join(q).join(q1)\
.join(p).join(p1)\
.join(D).join(D1)\
.join(k).join(k1)\
.join(l).join(l1)\
.join(m).join(m1)\
.join(z).join(z1)\
.join(w).join(w1)\
.join(pm).join(pm1)\
.join(WL).join(WH)

df=df.loc[df['time']!=0]	# Drop t=0 

# Save
df.to_csv("df.csv",index=False)
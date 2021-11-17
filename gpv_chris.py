# Import packages
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load and clean data
os.chdir("/Users/christophersaw/Desktop/GPV")
df=pd.read_csv(r'PS3Data.csv')
bidder1=df['Bidder 1']
bidder2=df['Bidder 2']

# Plot histograms
# bidder1.plot.hist(bins=100)
# bidder2.plot.hist(bins=100)

# Assume that bidders are iid
bids=pd.melt(df,value_vars=['Bidder 1', 'Bidder 2'])
b=bids['value'].to_numpy()

# Parameters
n=1000 # observations of iid bids from 2 symmetric bidders over 500 games
k=(2*math.pi)**(-0.5)

# Kernel for b
h_bids=1.06*np.std(b)*n**(-0.2) 

# Nonparametric CDF: G(b)
def G(X,N):
	one=np.ones(N)
	zero=np.zeros(N)
	kcdf_bids = (1/N) * np.sum( np.where(b<=X, one, zero ) )
	return kcdf_bids

# Nonparametric pdf: g(b)
def g(X,N,H):
	kpdf_bids = (k/(N*H)) * np.sum( np.exp( (-0.5)*((b-X)/H)**2 ) )
	return kpdf_bids

# Recover psudo-sample of v
# In 2-bidder auctions, v = b + G(b)/g(b)
v=np.zeros(n)
for i in range(n):
	v[i]=b[i]+G(b[i],n)/g(b[i],n,h_bids)


# Kernel for v
h_values=1.06*np.std(v)*n**(-0.2)

# Fit KDE and plot f(v)
sns.distplot(v, hist=False, kde=True, bins=int(n*h_values), color='black')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# Can also do a point-by-point plot

# Define function for F(v)
def F(Y,N):
	one=np.ones(N)
	zero=np.zeros(N)
	kcdf_values = (1/N) * np.sum( np.where(v<=Y, one, zero ) )
	return kcdf_values

# Define function for f(v)
def f(Y,N,H):
	kpdf_values = (k/(N*H)) * np.sum( np.exp( (-0.5)*((v-Y)/H)**2 ) )
	return kpdf_values


fv=np.zeros(n)
for i in range(n):
	fv[i]=f(v[i],n,h_values)

plt.scatter(v, fv)
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
import os
os.chdir("/Users/christophersaw/Desktop/blp")
import pandas as pd
import numpy as np
df=pd.read_csv(r'OTC_Data.csv',sep='\t')
df=df.rename(columns={'sales_': 'sales', 'price_': 'price', 'prom_': 'prom','cost_': 'cost'})

# Re-weight sales, costs and prices to 50tab package
df.loc[df['brand'].isin([1,4,7]), 'weight']=0.5
df.loc[df['brand'].isin([2,5,8,10]), 'weight']=1
df.loc[df['brand'].isin([3,6,9,11]), 'weight']=2
df['sales']=df['sales']*df['weight']
df['price']=df['price']/df['weight']
df['cost']=df['cost']/df['weight']

# Create categories/dummies for brand, store-brand and branded_product
df['branded_product']=0
df.loc[df['brand'].isin([1,2,3,4,5,6,7,8,9]), 'branded_product']=1

# Market ids, market shares
df['market']=df.groupby(['store','week']).ngroup()								# numeric identifier
df['market_ids']=df['store'].astype(str)+str('x')+df['week'].astype(str)		# string identifier
df['shares']=df['sales']/df['count']
 
# Calculate inside and outside shares
df['insideshare']=df.groupby(['market'])['shares'].transform('sum')				# checked that insideshares are between 0 and 1
df['outsideshare']=df['insideshare'].apply(lambda x: 1-x)

# Merge instruments
df2=pd.read_csv(r'OTCDataInstruments.csv',sep='\t')
data=pd.merge(df, df2, on=['store','week','brand']) # this is a one-one merge
data=data.sort_values(by=['store','week','brand'])

# Re-weight hausman prices
data['pricestore1']=data['pricestore1']/data['weight']
data['pricestore2']=data['pricestore2']/data['weight']
data['pricestore3']=data['pricestore3']/data['weight']
data['pricestore4']=data['pricestore4']/data['weight']
data['pricestore5']=data['pricestore5']/data['weight']
data['pricestore6']=data['pricestore6']/data['weight']
data['pricestore7']=data['pricestore7']/data['weight']
data['pricestore8']=data['pricestore8']/data['weight']
data['pricestore9']=data['pricestore9']/data['weight']
data['pricestore10']=data['pricestore10']/data['weight']
data['pricestore11']=data['pricestore11']/data['weight']
data['pricestore12']=data['pricestore12']/data['weight']
data['pricestore13']=data['pricestore13']/data['weight']
data['pricestore14']=data['pricestore14']/data['weight']
data['pricestore15']=data['pricestore15']/data['weight']
data['pricestore16']=data['pricestore16']/data['weight']
data['pricestore17']=data['pricestore17']/data['weight']
data['pricestore18']=data['pricestore18']/data['weight']
data['pricestore19']=data['pricestore19']/data['weight']
data['pricestore20']=data['pricestore20']/data['weight']
data['pricestore21']=data['pricestore21']/data['weight']
data['pricestore22']=data['pricestore22']/data['weight']
data['pricestore23']=data['pricestore23']/data['weight']
data['pricestore24']=data['pricestore24']/data['weight']
data['pricestore25']=data['pricestore25']/data['weight']
data['pricestore26']=data['pricestore26']/data['weight']
data['pricestore27']=data['pricestore27']/data['weight']
data['pricestore28']=data['pricestore28']/data['weight']
data['pricestore29']=data['pricestore29']/data['weight']
data['pricestore30']=data['pricestore30']/data['weight']

# Save data for BLP model
data.sort_values(by=['store','week']).to_csv('headache.csv',index=False)

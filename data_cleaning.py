import os
os.chdir("/Users/christophersaw/Desktop/blp")
import pandas as pd
import numpy as np
df=pd.read_csv(r'OTC_Data.csv',sep='\t')
df.columns=df.columns.str.replace('_','')

# Re-weight sales, costs and prices to 50tab package
df.loc[df['brand'].isin([1,4,7]), 'weight']		=	0.5
df.loc[df['brand'].isin([2,5,8,10]), 'weight']	=	1
df.loc[df['brand'].isin([3,6,9,11]), 'weight']	=	2
df['sales']	= df['sales']*df['weight']
df['price']	= df['price']/df['weight']
df['cost']	= df['cost']/df['weight']

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

# Save data for BLP model
df.sort_values(by=['store','week']).to_csv('headache.csv',index=False)

# Instruments
df2=pd.read_csv(r'OTCDataInstruments.csv',sep='\t')
df2=df2.sort_values(by=['store','week','brand'])

# Re-weight Hausman Prices
df2.columns=df2.columns.str.replace('_','')
df2.loc[df2['brand'].isin([1,4,7]), 'weight']		=	0.5
df2.loc[df2['brand'].isin([2,5,8,10]), 'weight']	=	1
df2.loc[df2['brand'].isin([3,6,9,11]), 'weight']	=	2
df2['cost']=df2['cost']/df2['weight']
df2['pricestore1']=df2['pricestore1']/df2['weight']
df2['pricestore2']=df2['pricestore2']/df2['weight']
df2['pricestore3']=df2['pricestore3']/df2['weight']
df2['pricestore4']=df2['pricestore4']/df2['weight']
df2['pricestore5']=df2['pricestore5']/df2['weight']
df2['pricestore6']=df2['pricestore6']/df2['weight']
df2['pricestore7']=df2['pricestore7']/df2['weight']
df2['pricestore8']=df2['pricestore8']/df2['weight']
df2['pricestore9']=df2['pricestore9']/df2['weight']
df2['pricestore10']=df2['pricestore10']/df2['weight']
df2['pricestore11']=df2['pricestore11']/df2['weight']
df2['pricestore12']=df2['pricestore12']/df2['weight']
df2['pricestore13']=df2['pricestore13']/df2['weight']
df2['pricestore14']=df2['pricestore14']/df2['weight']
df2['pricestore15']=df2['pricestore15']/df2['weight']
df2['pricestore16']=df2['pricestore16']/df2['weight']
df2['pricestore17']=df2['pricestore17']/df2['weight']
df2['pricestore18']=df2['pricestore18']/df2['weight']
df2['pricestore19']=df2['pricestore19']/df2['weight']
df2['pricestore20']=df2['pricestore20']/df2['weight']
df2['pricestore21']=df2['pricestore21']/df2['weight']
df2['pricestore22']=df2['pricestore22']/df2['weight']
df2['pricestore23']=df2['pricestore23']/df2['weight']
df2['pricestore24']=df2['pricestore24']/df2['weight']
df2['pricestore25']=df2['pricestore25']/df2['weight']
df2['pricestore26']=df2['pricestore26']/df2['weight']
df2['pricestore27']=df2['pricestore27']/df2['weight']
df2['pricestore28']=df2['pricestore28']/df2['weight']
df2['pricestore29']=df2['pricestore29']/df2['weight']
df2['pricestore30']=df2['pricestore30']/df2['weight']

# Save instruments for BLP model
df2.sort_values(by=['store','week']).to_csv('headache_instr.csv',index=False)
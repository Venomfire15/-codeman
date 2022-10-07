# -codeman
Vcet hackathon


#importing th required libraries abd modules

import matplotlib
from mistune import InlineParser
import pandas as pd

#reading the data from the dataset 
df = pd.read_csv('crop_production.csv',na_values='=')
#print(df)

#dropping the yeild and the its axis

df = df.drop('Yield',axis=1)
#df.info()

df = df[df['State_Name']=='Maharashtra']
#df.info()

df.isnull().sum()
df.head(6)

#importing libraries for the graphical representation graphical 
import matplotlib.pyplot as plt
import seaborn as sb

c_mat = df.corr()
fig = plt.figure(figsize = (15,15))
#heatmap method  and paramseters
sb.heatmap(c_mat, vmax= .8, square = True)
#display of heatmap
plt.show()

df = df[df['Crop_Year']>=2004]
#print(df)

#df.info()
#bulilding dummies 
df = df.join(pd.get_dummies(df['District_Name']))
df = df.join(pd.get_dummies(df['Season']))
df = df.join(pd.get_dummies(df['Crop']))
df = df.join(pd.get_dummies(df['State_Name']))
#print(df)

df['Yield'] = df['Production']/df['Area']
#print(df)

df = df.drop('Production', axis=1)
df = df.drop('District_Name',axis=1)
df = df.drop('State_Name',axis=1)
df = df.drop('Crop',axis=1)
df = df.drop('Season',axis=1)
#print(df)

#importing for the machine learning 
from sklearn import preprocessing

x = df[['Area']].values.astype(float)

#tools
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled

df['Area'] = x_scaled
#print(df)

df.head()

df = df.fillna(df.mean())

from sklearn.model_selection import train_test_split
a = df
b = df['Yield']
c = df.drop('Unnamed: 7',axis=1)

a = c.drop('Yield',axis=1)
d=len(a.columns)
#print(d)

#print(a.columns)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=42)
'''print(a_train)
print(a_test)
print(b_train)
print(b_test)'''

#importing mathematical tools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#%matplotlib inline
#learing and testing 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a_train = sc.fit_transform(a_train)
a_test = sc.transform(a_test)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(a_train, b_train)
b_pred = regr.predict(a_test)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
#printing the result after the test
print('MSE =', mse(b_pred, b_test))
print('MAE =', mae(b_pred, b_test))
print('R2 Score =', r2_score(b_pred, b_test))

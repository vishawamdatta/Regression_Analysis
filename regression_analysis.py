# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:02:46 2020

@author: VD
"""

## 1. OLS regression coefficients 
from sklearn import preprocessing

from sklearn.datasets import load_boston
import pandas as pd 
house_price = load_boston()
house_price.data = preprocessing.scale(house_price.data)
df = pd.DataFrame(house_price.data,
columns=house_price.feature_names)
df2=house_price.target;
from sklearn import linear_model
from sklearn.model_selection import train_test_split
l_reg=linear_model.LinearRegression();

x_train,x_test,y_train,y_test=train_test_split(df,df2,test_size=0.3)
l_reg.fit(x_train,y_train)
l_reg.score(df,df2)


from matplotlib import pyplot as plt
z=l_reg.coef_
name=house_price.feature_names
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,z)
ax.set_title("OLS Coefficients", fontsize=18)
plt.show()

##2. Plotting the ridge regression coefficients by varying lambda 

import numpy as np;
arr=np.empty((0,13))
from sklearn import datasets
boston = datasets.load_boston()
boston.data = preprocessing.scale(boston.data)
x=boston.data
y=boston.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

for i in range(201):
    reg = linear_model.Ridge(alpha=(i))
    reg.fit(x_train,y_train)
    arr=np.append(arr,[reg.coef_],axis=0)


fig, ax1 = plt.subplots()
ax1.plot(arr[:,5], 'g', arr[:,1], 'b', arr[:,8], 'r', arr[:,0], 'y', arr[:,9], 'c')
ax1.legend(labels=["Room",'Residential areas','Highway access','Crime Rate','Tax'])  
ax1.grid(True)
ax1.set_xlabel("Lambda Values")
ax1.set_ylabel("Beta Values")
ax1.set_title("Ridge Regression Coefficients", fontsize=18)
plt.show()


##3  Plotting the lasso regression coefficients by varying lambda 

arr2=np.empty((0,13))
for i in range(201):
    las = linear_model.Lasso(alpha=i)
    las.fit(x_train,y_train)
    arr2=np.append(arr2,[las.coef_],axis=0)


fig, ax2 = plt.subplots()
ax2.plot(arr2[:,5], 'g', arr2[:,1], 'b', arr2[:,8], 'r', arr2[:,0], 'y', arr2[:,9], 'c')
ax2.legend(labels=["Room",'Residential areas','Highway access','Crime Rate','Tax'])  
ax2.grid(True)
ax2.set_xlabel("Lambda Values")
ax2.set_ylabel("Beta Values")
ax2.set_title("Lasso Regression Coefficients lamda(0-200)", fontsize=18)
plt.show()



arr2=np.empty((0,13))
for i in range(201):
    las = linear_model.Lasso(alpha=(i)*0.05)
    las.fit(x_train,y_train)
    arr2=np.append(arr2,[las.coef_],axis=0)

k=np.arange(-1.25,11,1.25)
print(k)
fig, ax2 = plt.subplots()
ax2.plot(arr2[:,5], 'g', arr2[:,1], 'b', arr2[:,8], 'r', arr2[:,0], 'y', arr2[:,9], 'c')
ax2.legend(labels=["Room",'Residential areas','Highway access','Crime Rate','Tax'])  
ax2.grid(True)
ax2.set_xticklabels(k)
ax2.set_xlabel("Lambda Values")
ax2.set_ylabel("Beta Values")
ax2.set_title("Lasso Regression Coefficients lamda(0-10)", fontsize=18)
plt.show()

arr2=np.empty((0,13))
for i in range(201):
    las = linear_model.Lasso(alpha=(i)*0.005)
    las.fit(x_train,y_train)
    arr2=np.append(arr2,[las.coef_],axis=0)

k=np.arange(-.125,11,.125)
print(k)
fig, ax2 = plt.subplots()
ax2.plot(arr2[:,5], 'g', arr2[:,1], 'b', arr2[:,8], 'r', arr2[:,0], 'y', arr2[:,9], 'c')
ax2.legend(labels=["Room",'Residential areas','Highway access','Crime Rate','Tax'])  
ax2.grid(True)
ax2.set_xticklabels(k)
ax2.set_xlabel("Lambda Values")
ax2.set_ylabel("Beta Values")
ax2.set_title("Lasso Regression Coefficients lamda(0-1)", fontsize=18)
plt.show()


##4. Plotting the residuals 
from sklearn.metrics import mean_absolute_error
arr3=np.empty((0,2))

##graph for linear regresion
l_reg.fit(x_train,y_train)
linearpredict=l_reg.predict(x_train)
j=y_train-linearpredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.grid(True)
ax.set_title("Residuals for OLS", fontsize=18)
plt.show()
predict1=l_reg.predict(x_train)
predict2=l_reg.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)

##graph for ridge regresion
reg = linear_model.Ridge(alpha=10)
reg.fit(x_train,y_train)
ridgepredict=reg.predict(x_train)
j=y_train-ridgepredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.grid(True)
ax.set_title("Residuals for Ridge regression lamda=10", fontsize=18)
plt.show()

predict1=reg.predict(x_train)
predict2=reg.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)


reg = linear_model.Ridge(alpha=85)
reg.fit(x_train,y_train)
ridgepredict=reg.predict(x_train)
j=y_train-ridgepredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.set_title("Residuals for Ridge regression lamda=85", fontsize=18)
ax.grid(True)
plt.show()

predict1=reg.predict(x_train)
predict2=reg.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)


reg = linear_model.Ridge(alpha=150)
reg.fit(x_train,y_train)
ridgepredict=reg.predict(x_train)
j=y_train-ridgepredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.set_title("Residuals for Ridge regression lamda=150", fontsize=18)
ax.grid(True)
plt.show()

predict1=reg.predict(x_train)
predict2=reg.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)



## graph for lasso regression 
las = linear_model.Lasso(alpha=0.2)
las.fit(x_train,y_train)
laspredict=las.predict(x_train)
j=y_train-laspredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.set_title("Residuals for Lasso regression lamda=0.2", fontsize=18)
ax.grid(True)
plt.show()

predict1=las.predict(x_train)
predict2=las.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)


las = linear_model.Lasso(alpha=3)
las.fit(x_train,y_train)
laspredict=las.predict(x_train)
j=y_train-laspredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.grid(True)
ax.set_title("Residuals for Lasso regression lamda=3", fontsize=18)
plt.show()

predict1=las.predict(x_train)
predict2=las.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)



las = linear_model.Lasso(alpha=10)
las.fit(x_train,y_train)
laspredict=las.predict(x_train)
j=y_train-laspredict
name=[i for i in range(354)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(name,np.absolute(j))
ax.set_xlabel("Data points")
ax.set_ylabel("Residuals")
ax.set_title("Residuals for Lasso regression lamda=10", fontsize=18)
ax.grid(True)
plt.show()

predict1=las.predict(x_train)
predict2=las.predict(x_test)
temp2=mean_absolute_error(y_train, predict1)
temp3=mean_absolute_error(y_test,predict2)
temp=np.array([temp2,temp3])
arr3=np.append(arr3,[temp],axis=0)




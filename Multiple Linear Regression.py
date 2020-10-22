#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#importing datasets
dataset = pd.read_csv('50_Startups.csv')

#Extracting Independent and dependent Variable  
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 4].values
 
#Catgorical data  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# converting from matrix to array
x = np.array(ct.fit_transform(x)) 

# #avoiding the dummy variable trap: ---- Removing one dummy variable 
x=x[:, 1:]

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)


#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  


#Predicting the Test set result;  
y_pred= regressor.predict(x_test)  


df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)

# trying to predict the first x_test variable
pred= regressor.predict([[1.0,0.0,66051.52,182645.56,118148.2]]).reshape(-1,1)
print("I predict for you : ", pred)


#Checking the score  
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))  

  


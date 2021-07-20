##Importing important libraries---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##imprting Dataset-
path =  "http://bit.ly/w-data"
Data = pd.read_csv(path)
print("Data is successfully imported")

print(Data.head())
print(Data.describe())

print(Data.info())


##ploting Scatter plot----
plt.xlabel('Hours', fontsize=15)
plt.ylabel('Scores', fontsize=15)
plt.title('Hours studied vs Score', fontsize=10)
plt.scatter(Data.Hours, Data.Scores, color='blue', marker='*')
plt.show()


X = Data.iloc[:, :-1].values
Y = Data.iloc[:, 1].values
print(X)


print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

## We have Splitted Our Data Using 80:20 RULe(PARETO)
print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape  =", X_test.shape)
print("Y test.shape  =", Y_test.shape)


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()

linreg.fit(X_train,Y_train)
print("Training our algorithm is finished")

print("B0 =", linreg.intercept_, "\nB1 =", linreg.coef_)## β0 is Intercept & Slope of the line is β1.

##plotting the REGRESSION LINE---
Y0 = linreg.intercept_ + linreg.coef_*X_train

##plotting on train data
plt.scatter(X_train, Y_train, color='green', marker='+')
plt.plot(X_train, Y0, color='orange')
plt.xlabel("Hours", fontsize=15)
plt.ylabel("Scores", fontsize=15)
plt.title("Regression line(Train set)", fontsize=10)
plt.show()

Y_pred=linreg.predict(X_test)##predicting the Scores for test data
print(Y_pred)

#now print the Y_test.
print(Y_test)

#plotting line on test data
plt.plot(X_test, Y_pred, color='red')
plt.scatter(X_test, Y_test, color='black', marker='+')
plt.xlabel("Hours", fontsize=15)
plt.ylabel("Scores", fontsize=15)
plt.title("Regression line(Test set)", fontsize=10)
plt.show()

Y_test1 = list(Y_test)
prediction = list(Y_pred)
df_compare = pd.DataFrame({'Actual': Y_test1, 'Result': prediction})
print(df_compare)


from sklearn import metrics
metrics.r2_score(Y_test, Y_pred)##Goodness of fit Test

from sklearn.metrics import mean_squared_error, mean_absolute_error
MSE = metrics.mean_squared_error(Y_test, Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print("Mean Squared Error      = ", MSE)
print("Root Mean Squared Error = ", root_E)
print("Mean Absolute Error     = ", Abs_E)

Prediction_score = linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :", Prediction_score)
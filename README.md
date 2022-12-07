### Saud's-Portfolio
Data Science Portfolio Example

### Multiple Linear Regression Model
Here is a multiple linear regression model I worked on which was implemented using Scikit-Learn, which is one of the most popular
machine learning libraries for Python. The dataset is "LaptopSales.csv". We make use of features of laptops to
predict their sale price.

### Data Dictinary
* Configuration Index: a score based on laptop performance
* RAM (GB):Random Access Memory
* Processor Speeds (GHz)
* Integrated Wireless?
* HD Size (GB): hard drive size
* Bundled Applications: a set of single software programs that are sold together. Common types of bundled software include operating systems, utilities and accessories sold with desktop or laptop computers


### Read data into jupyter notebook
import pandas as pd

### Define file path and view the first 6 rows.
sales = pd.read_csv("C:/Users/Mohammad/Downloads/LaptopSales.csv")

sales.head(6)

![image](https://user-images.githubusercontent.com/63278449/206063384-24f2616a-fc5a-4a6a-8832-fe211bdf6e66.png)
![image](https://user-images.githubusercontent.com/63278449/206063419-373807d1-55f0-40b0-b4e4-5fc71c0fca82.png)


### View data type of variables
sales.dtypes

![image](https://user-images.githubusercontent.com/63278449/206063484-0124a981-ba52-4882-8883-9066e2c1daca.png)

### View summary statistics for numerical variables in the dataframe
sales.describe()

![image](https://user-images.githubusercontent.com/63278449/206057805-d2541d52-2785-4001-b5bd-01d78f1c4f6c.png)


### We check for NaN values (missing values) in this dataset. All the columns return False value indicating that there are no missing values in this dataset.

sales.isnull().any()

![image](https://user-images.githubusercontent.com/63278449/206063614-734772b7-f983-4659-ae61-453e3471ad64.png)
![image](https://user-images.githubusercontent.com/63278449/206063672-6bdfc93c-f9d4-44cd-ae29-53394d9c4f0c.png)


### Correlation chart to show our correlated values
corr = sales.corr()

corr

![image](https://user-images.githubusercontent.com/63278449/206058667-32ebb440-a4b9-44e7-a1d8-06d117c02b56.png)

### Creating Heat Map to show our correlated values by using seaborn and matplot libraries.
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

sns.heatmap(corr, annot = True,cmap = sns.diverging_palette(20, 220, n=200))

plt.title('Correlation Heatmap')

Text(0.5, 1, 'Correlation Heatmap')

![image](https://user-images.githubusercontent.com/63278449/206059429-d4c901f0-1364-4faa-a9a6-71ab8aa41ff7.png)

### Next step is to divide the data into “attributes” and “labels”. X variable contains all the attributes/features and y variable contains the target variable.

X = pd.get_dummies(sales[['Configuration Index','RAM (GB)', 'Processor Speeds (GHz)', 'Integrated Wireless',
'HD Size (GB)', 'Bundled Applications']], drop_first = True)
y = sales[['Retail Price']]

X

![image](https://user-images.githubusercontent.com/63278449/206059786-61d5a9c0-7876-475e-95b6-ce2137dd6c6b.png)
![image](https://user-images.githubusercontent.com/63278449/206059828-2743efd2-1f83-4060-97c9-44d9637c72a6.png)

### Now we split the data into training set and test set. We use 80% of the data as the training set and the rest 20% of the data as test set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape

(20519, 6)

### Now we are ready to train a linear regression model on the training set and estimate the coefficients.
from sklearn.linear_model import LinearRegression

sales_regressor = LinearRegression()

sales_regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(sales_regressor.coef_, columns= X.columns)

coeff_df

![image](https://user-images.githubusercontent.com/63278449/206060168-c5de5a44-8eab-4b37-8be4-62d811d7c936.png)

### Statsmodels fits a line passing through the origin, it doesn't fit an intercept. Hence we use the command 'ad
### Statsmodels however provides a convenience function called add_constant that adds a constant column to input data set.
import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)

est = sm.OLS(y_train, X_train2)

est2 = est.fit()

print(est2.summary())

![image](https://user-images.githubusercontent.com/63278449/206060459-faea549f-a720-4a4a-b8cd-d524ac90dbbf.png)


### Evaluating the linear regression model

y_train_pred = sales_regressor.predict(X_train)

y_test_pred = sales_regressor.predict(X_test)

### Scatterplot for Residial and Predicted Values
plt.figure(figsize = (15,8))

plt.scatter(y_train_pred, y_train_pred-y_train, c='steelblue', marker = 'o', label = "Training data")

plt.scatter(y_test_pred, y_test_pred-y_test, color = 'limegreen', marker = 's',label='Test data')

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.legend(loc = 'upper right')

plt.hlines(y = 0,xmin = 200, xmax = 700, color = 'black', lw =2)

plt.show()

![image](https://user-images.githubusercontent.com/63278449/206060808-9d8c3b4a-ce76-461f-a1f1-6b2eba3f7523.png)


### For regression algorithms, we can use three evaluation metrics to evaluate model performance:
1. Mean Squared Error (MSE) is the mean of the squared errors
2. Mean Absolute Error (MAE) is the mean of the absolute value of the errors.
3. Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors

from sklearn import metrics
import numpy as np

print('Mean Absolute Error on Test Data:', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Absolute Error on Training Data:', metrics.mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error on Test Data:', metrics.mean_squared_error(y_test, y_test_pred))

print('Mean Squared Error on Training Data:', metrics.mean_squared_error(y_train, y_train_pred))

print('Root Mean Squared Error on Test Data:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

print('Root Mean Squared Error on Training Data:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

### Regression for both Test & Training predictions
![image](https://user-images.githubusercontent.com/63278449/206066549-60893953-dd4c-445d-94bb-b3bed7d098c0.png)

### R squared for both Test & Training predictions
from sklearn.metrics import r2_score

r_sq_test = r2_score(y_test, y_test_pred)

r_sq_train = r2_score(y_train, y_train_pred)

print('R squared on test set:', r_sq_test)

print('R squared on training set:', r_sq_train)

![image](https://user-images.githubusercontent.com/63278449/206066732-0c3b3358-772b-40b7-8cee-e8a2e60f65dd.png)

### Based on your final analysis, we can conclude both test set and training set performed almost equally well since R squared is almost the same.





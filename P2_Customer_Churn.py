# Neo Company customer churn analysis python capstone project.
# Author: Charudatta Chousalkar

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import sweetviz as sv


customer = pd.read_csv("C:/Users/cchousal/Documents/Data Science/FINAL PROJECT WORK/PYTHON CERTIFICATION/PROJECT-3-8211-CAPSTONE-PROJECT-27JUN2020133817/Capstone-project-2/P3_customer_churn.csv")
print(customer.head())

## Analyzing and Displaying dataset using sweetviz
#cust_report = sv.analyze(customer)
#cust_report.show_html('Customer_Churn.html') 

# Data Manipulation

customer_5 = customer.iloc[:,4]

customer_15 = customer.iloc[:,14]

sme = (customer['gender'] == "Male") & (customer['SeniorCitizen'] == 1) & (customer['PaymentMethod'] == "Electronic check")
senior_male_electronic = customer[sme]
print(len(senior_male_electronic))

ctt = (customer['tenure'] > 70) | (customer['MonthlyCharges'] > 100)
customer_total_tenure = customer[ctt]
print(len(customer_total_tenure))

tmy = (customer['Contract'] == "Two year") & (customer['PaymentMethod'] == "Mailed check") & (customer['Churn'] == "Yes")
two_mail_yes = customer[tmy]
print(two_mail_yes.head())

customer_333 = customer.sample(n=333)
print(customer_333.head())

print('Churn_Distribution:\n',customer['Churn'].value_counts())

#*********************************************************************************************************************************************************************************************************

# Data Visualization

fig1 = plt.figure()
keys = customer['InternetService'].value_counts().keys().tolist()
values = customer['InternetService'].value_counts().tolist()
plt.bar(keys,values,color = 'orange')
plt.title('Distribution of Internet Services.')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
#plt.show()
fig1.savefig('C:/Users/cchousal/Documents/Data Science/FINAL PROJECT WORK/PYTHON CERTIFICATION/PROJECT-3-8211-CAPSTONE-PROJECT-27JUN2020133817/Capstone-project-2/InternetService_Bar_Plot.png')

fig2 = plt.figure()
plt.hist(customer['tenure'],bins=30,color='green')
plt.title('Distribution of Tenure')
#plt.show()
fig2.savefig('C:/Users/cchousal/Documents/Data Science/FINAL PROJECT WORK/PYTHON CERTIFICATION/PROJECT-3-8211-CAPSTONE-PROJECT-27JUN2020133817/Capstone-project-2/Tenure_Histogram_Plot.png')

fig3 = plt.figure()
x = customer['tenure']
y = customer['MonthlyCharges']
plt.scatter(x,y,color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')
#plt.show()
fig3.savefig('C:/Users/cchousal/Documents/Data Science/FINAL PROJECT WORK/PYTHON CERTIFICATION/PROJECT-3-8211-CAPSTONE-PROJECT-27JUN2020133817/Capstone-project-2/TenureVsMonthlyCharges_Scatter_Plot.png')

fig4 = plt.figure()
sns.boxplot(x='Contract', y='tenure', data=customer)
#plt.show()
fig4.savefig('C:/Users/cchousal/Documents/Data Science/FINAL PROJECT WORK/PYTHON CERTIFICATION/PROJECT-3-8211-CAPSTONE-PROJECT-27JUN2020133817/Capstone-project-2/ContractVsTenure_Box_Plot.png')

#*********************************************************************************************************************************************************************************************************

print('\nLinear Regression ::::\n')

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Check data
#print(customer.describe())
customer.plot(x='tenure',y='MonthlyCharges', style='o')
#plt.show()

#Define dependent and independent variables
x = customer[['tenure']]
y = customer[['MonthlyCharges']]

#Divide data in train test datasets  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#Model Building on train dataset.
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print('intercept =',regressor.intercept_,'coefficient =',regressor.coef_) #y=mx+c(m=coeff,c=intercept)

#Predict values based on test dataset.
y_pred = regressor.predict(x_test)

#Compare y_test and y_pred
error = np.sqrt(mean_squared_error(y_test,y_pred))
print('error =',error)

print('\nSimple Logistic Regression ::::\n')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

x = customer[['MonthlyCharges']]
y = customer[['Churn']]

#Divide data in train test datasets  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

log_model = LogisticRegression()
log_model.fit(x_train,y_train.values.ravel())

y_pred = log_model.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm,'\naccuracy =',acc)

print('\nMultiple Logistic Regression ::::\n')

x = customer[['MonthlyCharges','tenure']]
y = customer[['Churn']]

#Divide data in train test datasets  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

log_model = LogisticRegression()
log_model.fit(x_train,y_train.values.ravel())

y_pred = log_model.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm,'\naccuracy =',acc)

print('\nDecision Tree ::::\n')

from sklearn.tree import DecisionTreeClassifier

x = customer[['tenure']]
y = customer[['Churn']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

my_tree = DecisionTreeClassifier()
my_tree.fit(x_train,y_train)

y_pred = my_tree.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm,'\naccuracy =',acc)

print('\nRandom Forest ::::\n')

from sklearn.ensemble import RandomForestClassifier

x = customer[['tenure','MonthlyCharges']]
y = customer[['Churn']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

my_forest = RandomForestClassifier()
my_forest.fit(x_train,y_train.values.ravel())

y_pred = my_forest.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm,'\naccuracy =',acc)


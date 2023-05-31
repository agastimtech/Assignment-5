import numpy as np
import pandas as pd
from numpy import asarray

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve

import pickle
import asyncio

# 1. Problem Statement
# In this task we have to Predict retention of an employee within an organization
# such that whether the employee will leave the company or continue with it. An organization is only ]
# as good as its employees, and these people are the true source of its competitive advantage.
# Dependant variable : satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, Department, salary
# Independant variable : left

# Data Gathering
emp_df=pd.read_csv(r'/home/agasti/Desktop/Assignment_5/app/data.csv')
print(emp_df)

#EDA

print(emp_df.info())
print(emp_df.describe())

# satisfaction_level
plt.figure(figsize=(20,20))
sns.boxplot(emp_df['satisfaction_level'])
# plt.show()
print(emp_df['satisfaction_level'].skew())

plt.figure(figsize=(20,20))
sns.kdeplot(emp_df['satisfaction_level'])
# plt.show()

# last_evaluation
plt.figure(figsize=(20,20))
sns.boxplot(emp_df['last_evaluation'])
# plt.show()
print(emp_df['last_evaluation'].skew())

plt.figure(figsize=(20,20))
sns.kdeplot(emp_df['last_evaluation'])
# plt.show()

# number_project
print(emp_df['number_project'].value_counts())

#average_montly_hours
plt.figure(figsize=(20,20))
sns.boxplot(emp_df['average_montly_hours'])
# plt.show()
print(emp_df['average_montly_hours'].skew())

plt.figure(figsize=(20,20))
sns.kdeplot(emp_df['average_montly_hours'])
# plt.show()

# time_spend_company
print(emp_df['time_spend_company'].value_counts())

# Work_accident
print(emp_df['Work_accident'].value_counts())

# promotion_last_5years
print(emp_df['promotion_last_5years'].value_counts())

# Department
plt.figure(figsize=(20,20))
print(emp_df['Department'].value_counts(normalize=True).plot(kind='pie',autopct='%1.2f%%'))
# plt.show()
print(emp_df['Department'].value_counts())

# salary
print(emp_df['salary'].value_counts())

plt.figure(figsize=(20,20))
plt.xlabel('Salary')
plt.ylabel('Employees')
print(emp_df['salary'].value_counts().plot(kind='bar'))
# plt.show()

# left
print(emp_df['left'].value_counts(normalize=True)*100)

# plt.figure(figsize=(20,20))
# sns.pairplot(emp_df,hue='left')
# plt.show()

# Feature Engineering

emp_df['Department']=emp_df['Department'].replace({'sales':0,'technical':1,'support':2,'IT':3,
                                                   'product_mng':4,'marketing':5,'RandD':6,
                                                   'accounting':7,'hr':8,'management':9})
emp_df['salary'] = emp_df['salary'].replace({'low':0,'medium':1,'high':2})

print(emp_df['salary'].value_counts())

#Feature Selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

x = emp_df.drop('left',axis=1)
print(x)

# print(variance_inflation_factor(x.values,1))

vif_list = []
for i in range(x.shape[1]):
    vif = variance_inflation_factor(x.values,i)
    vif_list.append(round(vif,2))

s1 = pd.Series(vif_list,index=x.columns)
plt.figure(figsize=(20,20))
print(s1.sort_values().plot(kind='barh'))
# plt.show()

pd.set_option('display.max_columns', None)
print(emp_df.corr())

# Train Test Split
x = emp_df.drop('left',axis=1)
y = emp_df['left']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)

# print(y_test.value_counts(normalize=True))
# print(y_train.value_counts(normalize=True))

# ModelInstantiation
def model_building(algo,x,y):
    model = algo
    model.fit(x,y)
    return model

log_clf = model_building(LogisticRegression(),x_train,y_train)
print(log_clf)

def evaluate_model(model, ind_var, act):
    pred = model.predict(ind_var)

    acc_score = accuracy_score(act,pred)
    print('Accuracy Score : ',acc_score)
    print('***'*20)

    cnf_matrix = confusion_matrix(act, pred)
    print('Confusion Matrix : \n',cnf_matrix)
    print('***'*20)

    clf_report = classification_report(act,pred)
    print('Classification Report : \n',clf_report)
    print('***'*20)

    return pred

print('Testing Data Evaluation'.center(50,'*'))
y_pred_test=evaluate_model(log_clf,x_test,y_test)

print(y_test.value_counts())

standard_scaler = StandardScaler()

x_Train = standard_scaler.fit_transform(x_train[['satisfaction_level','last_evaluation','average_montly_hours']])
x_Train = pd.DataFrame(x_Train,columns=['satisfaction_level','last_evaluation','average_montly_hours'],index=x_train.index)
x_train_scaled = pd.concat([x_Train,x_train[['number_project','time_spend_company','Work_accident','promotion_last_5years','Department','salary']]],axis=1)
# print(x_train_scaled)

x_Test = standard_scaler.transform(x_test[['satisfaction_level','last_evaluation','average_montly_hours']])
x_Test = pd.DataFrame(x_Test,columns=['satisfaction_level','last_evaluation','average_montly_hours'],index=x_test.index)
x_test_scaled = pd.concat([x_Test,x_test[['number_project','time_spend_company','Work_accident','promotion_last_5years','Department','salary']]],axis=1)
# print(x_test_scaled)

log_clf_scaled = model_building(LogisticRegression(),x_train_scaled,y_train)
print(log_clf_scaled)

print('Testing Data Evaluation of Scaled'.center(50,'*'))
y_pred_test=evaluate_model(log_clf_scaled,x_test_scaled,y_test)

def save_model():
    # Code to save the model
    model_path = r'/home/agasti/Desktop/Assignment_5/app/model/model_pkl'
    pickle.dump(log_clf, open(model_path, 'wb'))
    return model_path

def load_model():
    # Load the saved model from the 'model' directory
    pickled_model = pickle.load(open(save_model(), 'rb'))
    return pickled_model

# print(x_test)
# a=log_clf.predict([[0.45,0.54,2,154,3,0,0,5,0]])
# print(a[0])
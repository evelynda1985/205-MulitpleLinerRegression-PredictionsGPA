#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


# In[15]:


raw_data = pd.read_csv('1.03. Dummies.csv')


# In[3]:


raw_data


# In[4]:


data = raw_data.copy()


# In[5]:


data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})


# In[6]:


data


# In[7]:


data.describe()


# In[8]:


## Regression


# In[9]:


y = data['GPA']
x1 = data[['SAT','Attendance']]


# In[10]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[11]:


plt.scatter(data['SAT'],y, c=data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.6439 + 0.2226 + 0.0014 * data['SAT']
yhat = 0.0017 * data['SAT'] + 0.275
fig = plt.plot(data['SAT'], yhat_no, lw=2, c='red', label='regression line 1')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='orange', label ='regression line 2')
fig = plt.plot(data['SAT'], yhat, lw=3, c='blue', label ='regressionline')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[17]:


x


# In[20]:


new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance':[0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data


# In[22]:


new_data.rename(index={0:'Bob', 1:'Alice'})


# In[24]:


predictions = results.predict(new_data)
predictions


# In[25]:


predictionsdf =  pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:'Bob', 1:'Alice'})


# In[ ]:





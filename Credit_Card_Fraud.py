#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().values.any()


# In[6]:


fraud=data[data['Class']==1]
genuine=data[data['Class']==0]
print("Fraud cases:",len(fraud))
print("Genuine cases:",len(genuine))


# In[7]:


fraud.Amount.describe()


# In[8]:


genuine.Amount.describe()


# In[9]:


data.groupby('Class').mean()


# In[10]:


genuine_sample=genuine.sample(n=492)


# In[11]:


new_data=pd.concat([genuine_sample,fraud],axis=0)


# In[12]:


new_data.head()


# In[13]:


new_data['Class'].value_counts()


# In[14]:


new_data.groupby('Class').mean()


# In[15]:


X=new_data.drop(columns='Class',axis=1)
Y=new_data['Class']


# In[16]:


print(X)
print(Y)


# In[17]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[18]:


print(X.shape,X_train.shape,X_test.shape)


# In[19]:


model=LogisticRegression()


# In[20]:


model.fit(X_train,Y_train)


# In[21]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:{}'.format(training_data_accuracy))


# In[22]:


X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on testing data:{}'.format(testing_data_accuracy))


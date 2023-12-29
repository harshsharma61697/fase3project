#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df=pd.read_csv('C_incom.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df['Income'].value_counts()


# In[9]:


df['Sex'].value_counts()


# In[10]:


df['Native_country'].value_counts()


# In[11]:


df['Workclass'].value_counts()


# In[12]:


df['Occupation'].value_counts()


# In[13]:


df['Education'].value_counts()


# In[14]:


df=df.drop(['Education','Fnlwgt'],axis=1)


# In[15]:


df.head(1)


# In[16]:


df.replace('?',np.NaN,inplace=True)


# In[17]:


df.fillna(method='ffill',inplace=True)

Encoding
# In[18]:


import sklearn
from sklearn.preprocessing import LabelEncoder


# In[19]:


le=LabelEncoder()
le


# In[20]:


df['Workclass']=le.fit_transform(df['Workclass'])
df['Marital_status']=le.fit_transform(df['Marital_status'])
df['Occupation']=le.fit_transform(df['Occupation'])
df['Relationship']=le.fit_transform(df['Relationship'])
df['Race']=le.fit_transform(df['Race'])
df['Sex']=le.fit_transform(df['Sex'])
df['Native_country']=le.fit_transform(df['Native_country'])
df['Income']=le.fit_transform(df['Income'])


# In[21]:


df.head()


# In[22]:


import seaborn as sns


# In[23]:


sns.countplot(x='Income',data=df)


# In[24]:


sns.countplot(x='Age',data=df)


# In[25]:


sns.barplot(x='Income',y='Age',data=df)


# In[26]:


sns.countplot(x='Sex',data=df)


# In[27]:


sns.barplot(x='Income',y="Sex",data=df)


# In[28]:


sns.heatmap(df.corr())


# In[29]:


sns.pairplot(df)


# In[42]:


x=df.drop(['Income'],axis=1) 
y=df['Income']


# In[43]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[55]:


from sklearn.naive_bayes import GaussianNB


# In[56]:


gb=GaussianNB()
gb


# In[57]:


gb.fit(x_train,y_train)


# In[58]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[59]:


predict=gb.predict(x_test)


# In[60]:


print(classification_report(y_test,predict))


# In[61]:


print(confusion_matrix(y_test,predict))


# In[62]:


print(accuracy_score(y_test,predict))


# In[ ]:





# In[ ]:





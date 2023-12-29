#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd 
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt 
import sklearn


# In[3]:


df=pd.read_csv('Rain_csv')


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.info()


# In[10]:


df=df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','Date','Location'],axis=1)


# In[11]:


df.head()


# In[12]:


df=df.dropna(axis=0)
df.shape


# In[14]:


df.columns


# In[20]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['WindGustDir']=lb.fit_transform(df['WindGustDir'])
df['WindDir9am']=lb.fit_transform(df['WindDir9am'])
df['WindDir3pm']=lb.fit_transform(df['WindDir3pm'])
df['RainToday']=lb.fit_transform(df['RainToday'])
df['RainTomorrow']=lb.fit_transform(df['RainTomorrow'])


# In[21]:


df.head(4)


# In[22]:


x=df.drop(['RainTomorrow'],axis=1)
y=df['RainTomorrow']


# In[23]:


x.head()


# In[25]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='MaxTemp',y="MinTemp",hue="RainTomorrow",palette='inferno',data=df)


# In[26]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='Humidity9am',y="Temp9am",hue="RainTomorrow",palette='inferno',data=df)


# In[27]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='Humidity3pm',y="Temp3pm",hue="RainTomorrow",palette='inferno',data=df)


# In[28]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='WindGustSpeed',y="WindDir9am",hue="RainTomorrow",palette='inferno',data=df)


# In[29]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr())


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[39]:


from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


lm=LogisticRegression()
lm


# In[43]:


lm.fit(x_train,y_train)


# In[44]:


prediction=lm.predict(x_test)


# In[45]:


print(confusion_matrix(y_test,prediction))


# In[46]:


print(accuracy_score(y_test,prediction))


# In[48]:


print(classification_report(y_test,prediction))


# In[50]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt


# In[51]:


dt.fit(x_test,y_test)


# In[53]:


pred=dt.predict(x_test)


# In[54]:


print(accuracy_score(y_test,pred))


# In[55]:


print(classification_report(y_test,pred))


# In[57]:


print(confusion_matrix(y_test,pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[3]:


df1=pd.read_csv("zm_csv")


# In[4]:


df1.head(5)


# In[5]:


len(df1.columns)


# In[6]:


df1['Rating text'].unique()


# In[7]:


df1.dtypes


# In[8]:


df1.isnull().sum()


# In[9]:


df1.info()


# In[10]:


df=df1.drop(['Restaurant ID','Restaurant Name','City','Address','Locality','Locality Verbose','Cuisines','Rating color','Switch to order menu'],axis=1)


# In[11]:


df


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


lb=LabelEncoder()
lb


# In[14]:


df['Currency']=lb.fit_transform(df['Currency'])
df['Has Table booking']=lb.fit_transform(df['Has Table booking'])
df['Has Online delivery']=lb.fit_transform(df['Has Online delivery'])
df['Is delivering now']=lb.fit_transform(df['Is delivering now'])
#df['Switch to order menu']=lb.fit_transform(df['Switch to order menu'])
df['Rating text']=lb.fit_transform(df['Rating text'])


# In[15]:


df


# In[16]:


x=df.drop(['Votes'],axis=1)
y=df['Votes']


# In[17]:


x.head()


# In[18]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Longitude',y='Latitude',hue='Rating text',palette='inferno',data=df)


# In[19]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Is delivering now',y='Price range',hue='Rating text',palette='inferno',data=df)


# In[20]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr())


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[23]:


from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


lr=LogisticRegression()
lr


# In[26]:


lr.fit(x_train,y_train)


# In[27]:


pred=lr.predict(x_test)


# In[28]:


print(confusion_matrix(y_test,pred))


# In[29]:


print(accuracy_score(y_test,pred))


# In[30]:


print(classification_report(y_test,pred))


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt


# In[32]:


dt.fit(x_test,y_test)


# In[33]:


pred=dt.predict(x_test)


# In[34]:


print(accuracy_score(y_test,pred))


# In[35]:


print(classification_report(y_test,pred))


# In[36]:


print(confusion_matrix(y_test,pred))


# In[43]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[37]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr


# In[38]:


lr.fit(x_test,y_test)


# In[41]:


pred=lr.predict(x_test)


# In[44]:


print(mean_squared_error(y_test,pred))


# In[45]:


print(mean_absolute_error(y_test,pred))


# In[46]:


lr.score(x_train,y_train)


# In[47]:


from sklearn.metrics import r2_score


# In[48]:


print(r2_score(y_test,pred))


# In[49]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn


# In[50]:


kn.fit(x_test,y_test)


# In[51]:


predict=kn.predict(x_test)


# In[52]:


print(confusion_matrix(y_test,predict))


# In[53]:


print(accuracy_score(y_test,predict))


# In[54]:


print(classification_report(y_test,predict))


# In[55]:


from sklearn.naive_bayes import GaussianNB


# In[56]:


gb=GaussianNB()
gb


# In[57]:


gb.fit(x_test,y_test)


# In[58]:


pre=gb.predict(x_test)


# In[59]:


print(confusion_matrix(y_test,pre))


# In[60]:


print(accuracy_score(y_test,pre))


# In[61]:


print(classification_report(y_test,pre))


# In[ ]:





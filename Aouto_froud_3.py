#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df=pd.read_csv("Aouto_froud.csv")


# In[3]:


df.head(4)


# In[4]:


df.shape


# In[5]:


df.replace('?',np.nan,inplace=True)


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df['collision_type']=df['collision_type'].fillna(df['collision_type'].mode()[0])


# In[10]:


df['property_damage']=df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available']=df['police_report_available'].fillna(df['police_report_available'].mode()[0])


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:



sns.heatmap(df.corr())


# In[13]:


sns.pairplot(df)


# In[14]:


sns.countplot(x='fraud_reported',data=df)


# In[15]:


sns.barplot(x='policy_number',y='fraud_reported',data=df)


# In[16]:


sns.barplot(x='age',y='fraud_reported',data=df)


# In[17]:


dro=['policy_bind_date','policy_state','policy_number','policy_state','auto_make','auto_model','auto_year','_c39','police_report_available']


# In[18]:


df.drop(dro,inplace=True,axis=1 )


# In[19]:


df


# In[20]:


sns.heatmap(df.corr())


# In[21]:


df.drop(columns=['age','total_claim_amount'],inplace=True,axis=1)


# In[22]:


df.head(2)


# In[23]:


x=df.drop('fraud_reported',axis=1)
y=df['fraud_reported']


# In[24]:


cat_columns=x.select_dtypes(include=['object'])
data=pd.get_dummies(cat_columns,drop_first=True)


# In[25]:


data.head(4)


# In[26]:


num_columns=x.select_dtypes(include=['int64'])
x=pd.concat([num_columns,data],axis=1)


# In[27]:


x.head(4)


# In[28]:


sns.boxplot(data=x)


# In[29]:


plt.figure(figsize=(20,15))
plotnumber =1

for  col in x.columns:
    if plotnumber==24:
        ax=plt.subplot(5,5,plotnumber)
        sns.boxplot(x[col])
        plt.xlabel(col,fontsize=15)
        
    plotnumber+=1
plt.tight_layout() 
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[34]:


x_train.head()


# In[40]:


numerical_data=x_train[['months_as_customer','policy_deductable','umbrella_limit','insured_zip','capital-gains','capital-loss','incident_hour_of_the_day','number_of_vehicles_involved','bodily_injuries','witnesses','incident_location_9879 Apache Drive','incident_location_9910 Maple Ave','incident_location_9911 Britain Lane','incident_location_9918 Andromedia Drive','incident_location_9929 Rock Drive','incident_location_9935 4th Drive','incident_location_9942 Tree Ave','incident_location_9980 Lincoln Ave','incident_location_9988 Rock Ridge','property_damage_YES']]


# In[37]:


from sklearn.preprocessing import StandardScaler


# In[39]:


ler=StandardScaler()
ler


# In[41]:


s_data=ler.fit_transform(numerical_data)


# In[43]:


df2=pd.DataFrame(data=s_data,columns=numerical_data.columns,index=x_train.index)


# In[44]:


df2


# In[45]:


from sklearn.svm import SVC


# In[56]:


svc=SVC()
svc.fit(x_train,y_train)
pred=svc.predict(x_test)


# In[57]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[58]:


train_acc=accuracy_score(y_train,svc.predict(x_train))
test_acc=accuracy_score(y_test,pred)


# In[59]:


print(train_acc)


# In[60]:


print(test_acc)


# In[62]:


print(classification_report(y_test,pred))


# In[63]:


print(confusion_matrix(y_test,pred))


# In[65]:


from sklearn.tree import DecisionTreeClassifier


# In[66]:


dt=DecisionTreeClassifier()


# In[68]:


dt.fit(x_train,y_train)
predt=dt.predict(x_test)


# In[69]:


print(accuracy_score(y_train,dt.predict(x_train)))


# In[70]:


print(accuracy_score(y_test,predt))


# In[75]:


print(confusion_matrix(y_test,predt))


# In[74]:


print(classification_report(y_test,predt))


# In[ ]:





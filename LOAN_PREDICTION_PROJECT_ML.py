#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dependencies
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score


# In[2]:


#Data collection and data preprocesssing
loan_dataset = pd.read_csv(r"C:\Users\DELL\Downloads\loan_data_set.csv")


# In[3]:


type(loan_dataset)


# In[4]:


#printing the first 5 rows of the dataframe:
loan_dataset.head()


# In[5]:


#total values in our dataset
loan_dataset.shape


# In[6]:


#statistical measures
loan_dataset.describe()


# In[7]:


#number of misssing values in each column:
loan_dataset.isnull().sum()


# In[8]:


#dropping the missing values:
loan_dataset=loan_dataset.dropna()


# In[9]:


loan_dataset.isnull().sum() #now we are having 0 null values in our dataset


# In[10]:


#label encoding
#making numerical value for alphabetical value
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[11]:


#again printing the first 5 rows of the dataset
loan_dataset.head() 
#missing value dataset is dropped from the dataset


# In[12]:


#dependent column values,will define type and amount of data 
loan_dataset['Dependents'].value_counts()


# In[13]:


#replacing the value of 3+ for 4(just a random value 4)
loan_dataset=loan_dataset.replace(to_replace="3+",value =4)


# In[14]:


loan_dataset['Dependents'].value_counts() #3+ is replaced with 4


# In[15]:


#DATA VISUALIZATION:
#education and loan status:
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)#0 for no and 1 for yes,more approval for the graduted one


# In[16]:


loan_dataset['Loan_Status'].value_counts() #total  entries in loan status


# In[17]:


#marital status & loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset) #more chances for the married one as both of them can contribute to pay back the amount of loan


# In[18]:


#we can similarly compare loan status with different aspects as per the requirement basis:


# In[19]:


#All the alphabetical enteries have beeen repalced by the numerical one:
loan_dataset.replace({"Married":{'No':0,'Yes':1}},inplace=True)
loan_dataset.replace({"Gender":{'Male':1,'Female':1}},inplace=True)
loan_dataset.replace({"Self_Employed":{'No':0,'Yes':1}},inplace=True)
loan_dataset.replace({"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)
loan_dataset.replace({"Education":{'Graduate':0,'Not Graduate':1}},inplace=True)
    
    
    


# In[65]:


loan_dataset.tail()#our new dataset with all numerical entries:data preproceessing


# In[21]:


#sepreating the data and label,whwn we remove a row axis is 0 and if a column it is 1
x= loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y= loan_dataset['Loan_Status']


# In[22]:


print(x)
print(y)


# In[23]:


#splitting the data into trainning and the testing data:
#TRAIN TEST SPLIT: FROM sklearn modelmselection modul
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)
#x represnt data and y represnt the labels
#0.1 means 10%
 #stratify means to have equal proportions of 1 and 0 in the test data) #random values split bthe  
    
    


# In[24]:


print(x.shape,x_train.shape,x_test.shape)


# In[ ]:





# In[25]:


#trainning the model:support vector model ,We are doing a classification task in which we divide whole in two sections either loan is approaved or rejected
#SVC IS SUPPORT VECTOR CLASSIFIER
classifier=svm.SVC(kernel='linear')#now classifier is a svm model now 


# In[26]:


#trainning for support vector model
classifier.fit(x_train,y_train)


# In[28]:


#model evalution
#accuracy score of the traiining data
x_train_prediction=classifier.predict(x_train)
trainning_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[29]:


print('accuracy on the trainning data:',trainning_data_accuracy)


# In[32]:


x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[33]:


print('accuracy on the test data:',test_data_accuracy)


# In[61]:


#avoid overfitting should be avoided:
#make a predictive system:
input_data=( 1 ,1 ,1, 0,0,4583,1508.0 ,188.0 ,360.0 ,1.0  ,2)

input_data_as_numpy_array =np.asarray(input_data)

input_data_r=input_data_as_numpy_array.reshape(1,-1)
prediction= classifier.predict(input_data_r)
print(prediction)


# In[51]:


if(prediction[0]==0):
    print("the person is not approved for the loan")
else:
    print("the person is approved for loan")


# In[67]:


input_data=( 1,2,0,0,7583,0.0,187.0,360.0,1.0,2,1)
input_data_as_numpy_array =np.asarray(input_data)

input_data_r=input_data_as_numpy_array.reshape(1,-1)
prediction= classifier.predict(input_data_r)
print(prediction)


# In[68]:


if(prediction[0]==0):
    print("the person is not approved for the loan")
else:
    print("the person is approved for loan")


# In[ ]:





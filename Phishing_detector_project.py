#!/usr/bin/env python
# coding: utf-8

# # Phishing Detector using LR

# The document has to specify the requirements for the project “Build a detector for Phishing websites (LR).” Apart from specifying the functional and non-functional requirements for the project, it also serves as an input for project scoping.
# 

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# ### Importing the dataset phishing.txt

# In[2]:


data = pd.read_csv('phishing.txt',header =None)


# ### Data understanding and Exploration

# In[3]:


data.head()


# In[4]:


data.shape


# ### Adding columns to the dataset

# In[5]:


data.columns = ['having_IP_Address','URL_length','Shortining_Service','having_At_Symbol',
              'double_slash_redirecting','Prefix_suffix','having_Sub_Domain','SSLfinal_State','Domain_registration_length',
               'Favion','Port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH',
               'Submitting_to_email','Abnormal_URL','Redirect','on_movesover','RightClick','PopUpwindow','IFrame',
                'age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index',
                'Links_pointing_to_page','Statistical_Report','Result']
        


# In[6]:


data.head()


# In[7]:


data.columns


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.corr()


# In[11]:


sns.heatmap(data.corr(),cmap ='viridis',linecolor = 'black')


# ### Splitting the dataset into independent variables and dependent variables

# In[12]:


X = data.iloc[:,0:30].values


# In[13]:


X.shape


# In[14]:


y = data.loc[:,['Result']].values


# In[15]:


y.shape


# ### Spitting the dataset into training and testing dataset with ratio 70:30 
70% of the dataset is goes for training
30% of the dataset is goes for testing
# In[16]:


# Importing the train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.30,random_state =1)


# In[17]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression Model

# In[18]:


# Importing the classifier from linear model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()


# In[19]:


classifier.get_params().keys()


# In[20]:


# applying grid search to find best performing parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.001,0.01,0.1,1,10,100,1000],
               'penalty':['l1','l2']
              }]
grid_search = GridSearchCV(classifier,parameters,cv=5,n_jobs =-1)
grid_search.fit(X_train,y_train)
# Printing best parameters
print('Best Accuracy =',(grid_search.best_score_))
print('Best parameters = ',(grid_search.best_params_))


# In[21]:


# Instantiate the classifier Logistic Regression
classifier = LogisticRegression(C=0.1,penalty ='l1')


# In[22]:


# Fitting the classifier or model on training dataset to train
classifier.fit(X_train,y_train)


# In[23]:


# Predicting the values on test dataset
y_pred = classifier.predict(X_test)


# In[24]:


# Confusion matrix for the LR classifier
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[25]:


TP = cm[0][0]
print('True Postive  = ',TP)
FP = cm[0][1]
print('False Postive = ',FP)
FN = cm[1][0]
print('False Negative = ',FN)
TN = cm[1][1]
print('True Negative = ',TN)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print ('The Accuracy of the LR model : ',round(accuracy*100,ndigits =2),'%')


# # Random Forest Classification Model

# In[89]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =700,
                                   criterion ='entropy',
                                   max_features ='sqrt',
                                   random_state=0)


# In[90]:


classifier.fit(X_train,y_train)


# In[91]:


y_pred = classifier.predict(X_test)


# In[92]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[93]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[ ]:





# # XGBoost Model

# In[32]:


get_ipython().run_line_magic('time', '')
# Importing the XGBoost model from scikit learn ensemble 
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimator =1000)
classifier.fit(X_train,y_train)


# In[33]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[34]:


# Confusion matrix for evaluation to get the accuracy of the XGBoost model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[35]:


sns.heatmap(cm,annot = True,cmap="cividis")


# In[36]:


# Accuracy of the XGBoost model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
model_accuracy


# # The Champion Model out of LR,RF and XGBoost is Random Forest of accuracy 96%

# # Exercies 2

# Train with only two input parameters - parameter Prefix_Suffix and 13 URL_of_Anchor.
# Check accuracy using the test data and compare the accuracy with the previous value.

# In[37]:


data.head()


# In[38]:


data.columns


# In[39]:


X1 = data.loc[:,['Prefix_suffix','URL_of_Anchor']].values


# In[40]:


y1 = data.loc[:,['Result']].values


# In[41]:


X = pd.DataFrame(X1)


# In[42]:


y = pd.DataFrame(y)


# # Data understanding and Exploration

# In[43]:


X.info()


# In[44]:


X.describe().transpose()


# In[45]:


sns.pairplot(X,diag_kind = 'kde')


# In[46]:


sns.heatmap(X.corr(),cmap = 'YlGnBu',annot =True)


# In[ ]:





# In[47]:


y1 = y.values


# ### Spitting the dataset into training and testing dataset with ratio 70:30 

# In[48]:


# Importing the train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size =0.30,random_state =1)


# In[49]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression Model

# In[66]:


# Importing the classifier from linear model 
from sklearn.linear_model import LogisticRegression


# In[67]:


# Instantiate the classifier Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[68]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[69]:


# Confusion matrix for evaluation to get the accuracy of the XGBoost model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[70]:


TP = cm[0][0]
print('True Postive  = ',TP)
FP = cm[0][1]
print('False Postive = ',FP)
FN = cm[1][0]
print('False Negative = ',FN)
TN = cm[1][1]
print('True Negative = ',TN)


# In[73]:


# Accuracy of the XGBoost model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
model_accuracy


# # Random Forest Classification Model

# In[72]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =700,
                                   criterion ='entropy',
                                   max_features ='sqrt',
                                   random_state=0)


# In[74]:


classifier.fit(X_train,y_train)


# In[75]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[76]:


# Confusion matrix for evaluation to get the accuracy of the XGBoost model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[77]:


TP = cm[0][0]
print('True Postive  = ',TP)
FP = cm[0][1]
print('False Postive = ',FP)
FN = cm[1][0]
print('False Negative = ',FN)
TN = cm[1][1]
print('True Negative = ',TN)


# In[94]:


# Accuracy of the XGBoost model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
model_accuracy


# # XGBoost Model

# In[79]:


get_ipython().run_line_magic('time', '')
# Importing the XGBoost model from scikit learn ensemble 
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimator =1000)
classifier.fit(X_train,y_train)


# In[80]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[81]:


# Confusion matrix for evaluation to get the accuracy of the XGBoost model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[96]:


# Accuracy of the XGBoost model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#notice: Disable all warnings 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# In[7]:


import os
os.chdir('C:\\Users\\Shubham\\Downloads')
df = pd.read_csv('loan_train.csv')
df.head()


# In[8]:


df.shape


# In[9]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[10]:


df['loan_status'].value_counts()


# In[11]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('pip install seaborn')


# In[12]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[13]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[14]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[15]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[16]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[17]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[18]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[19]:


df[['Principal','terms','age','Gender','education']].head()


# In[20]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[21]:


X = Feature
X[0:5]


# In[22]:


y = df['loan_status'].values
y[0:5]


# In[23]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[24]:


get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')
get_ipython().system('pip install six')


# In[26]:


import numpy as np 
import pandas as pd
import itertools
import pydotplus
from six import StringIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss


# # K NEAREST NEIGHBOR(KNN)

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[28]:


Ks = 10
mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

mean_acc


# In[73]:


k = 7
#Train Model and Predict  
neigh7 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print("Test set Accuracy: ", metrics.accuracy_score(y_test, neigh7.predict(X_test)))
#print("Test set Jaccard Score: ", metrics.jaccard_score(y_test, neigh7.predict(X_test),loan_status = 'PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(y_test, neigh7.predict(X_test), average='weighted'))


# # DECISION TREE

# In[74]:


from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = 3)
print("Train_set:", X_train.shape, y_train.shape)
print("Test_set:", X_test.shape, y_test.shape)

LoanTree = DecisionTreeClassifier(criterion="entropy")
#Fit the Model.
LoanTree.fit(X_train,y_train)
predTree = LoanTree.predict(X_test)


# In[75]:


y_pred = LoanTree.predict(X_test)
#Print predicted test features
print (y_pred[0:5])
print (y_test[0:5])


# In[76]:


print("Test set Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[77]:


DT_F1_Score = f1_score(y_test, y_pred, average='weighted')
DT_F1_Score


# In[78]:


#Jaccard Score

print("Test set Jaccard Score: ", metrics.jaccard_score(y_test, predTree,pos_label='PAIDOFF'))


# # SUPPORT VECTOR MACHINE

# In[79]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[80]:


yhat = clf.predict(X_test)
yhat [0:5]


# In[81]:


print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
print("Test set Jaccard Score: ", metrics.jaccard_score(y_test, yhat, pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(y_test, yhat, average='weighted'))


# # LOGISTICS REGRESSION

# In[82]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.1, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
print("Test set Jaccard Score: ", metrics.jaccard_score(y_test, yhat, pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(y_test, yhat, average='weighted'))
print("Test set Log Loss Score: ", metrics.log_loss(y_test, yhat_prob))


# # MODEL EVALUATION USING TEST SET

# In[83]:


#from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[84]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# # LOADING TEST SET

# In[85]:


os.chdir('C:\\Users\\Shubham\\Downloads')
test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[ ]:





# In[86]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_Feature.head()


# In[87]:


XX = test_Feature
yy = test_df['loan_status'].values

XX= preprocessing.StandardScaler().fit(XX).transform(XX)
XX[0:5]


# In[88]:


print("Test set Accuracy: ", metrics.accuracy_score(yy, neigh7.predict(XX)))
print("Test set Jaccard Score: ", metrics.jaccard_score(yy, neigh7.predict(XX),pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(yy, neigh7.predict(XX), average='weighted'))


# In[89]:


predTree = LoanTree.predict(XX)
print("Test set Accuracy: ", metrics.accuracy_score(yy, predTree))
print("Test set Jaccard Score: ", metrics.jaccard_score(yy, predTree,pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(yy, predTree, average='weighted'))


# In[90]:


yyhat = clf.predict(XX)
print("Test set Accuracy: ", metrics.accuracy_score(yy, yyhat))
print("Test set Jaccard Score: ", metrics.jaccard_score(yy, yyhat,pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(yy, yyhat, average='weighted'))


# In[91]:


yyhat = LR.predict(XX)
yyhat_prob = LR.predict_proba(XX)
print("Test set Accuracy: ", metrics.accuracy_score(yy, yyhat))
print("Test set Jaccard Score: ", metrics.jaccard_score(yy, yyhat, pos_label='PAIDOFF'))
print("Test set F1 Score: ", metrics.f1_score(yy, yyhat, average='weighted'))
print("Test set Log Loss Score: ", metrics.log_loss(yy, yyhat_prob))


# In[ ]:





# In[ ]:





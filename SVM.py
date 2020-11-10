#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cell_df = pd.read_csv('mydata.csv')
cell_df.dtypes




cell_df.columns
features_df = cell_df[['age','gender','height','weight','bplo','plus','smoking','cardio','active','alchol','gluc','cholestrol']]

X = np.asarray(features_df)
y = np.asarray(cell_df['bphi'])

y[0:5]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape
y_train.shape
X_test.shape
y_test.shape


from sklearn import svm
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[11]:


print ("Score:", classifier.score(X_test, y_test))
y_predict


# In[5]:


y_predict.shape


# In[ ]:





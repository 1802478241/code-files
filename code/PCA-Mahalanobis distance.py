
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.cross_decomposition import PLSRegression

import numpy as np 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_excel(r"_file_",index_col=None)#x

if __name__ == "__main__":
    data = np.array(df)

 
    sklearn_pca = PCA(n_components=10)
    data_2 = sklearn_pca.fit_transform(data)
    
from numpy import *

x=pd.DataFrame(data_2)
x1=mean(x)
M=np.shape(x)


import numpy as np

def mashi_distance(x,y):  
  m=x.shape[0]
  x=np.vstack([x,y])
  
 
  XT=x.T
  S=np.cov(XT)  
  
        
  SI = np.linalg.inv(S) 


  n=x.shape[0]

  d1=[]

  for i in range(0,n):

    

      delta=x[i]-x[M]
      d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))

      print(d)

      d1.append(d)
  return d1



x = x
y = x1.T
md = mashi_distance(x,y)




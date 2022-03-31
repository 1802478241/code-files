
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas  as pd


from __future__ import division
from __future__ import print_function

import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

   
    X_train, y_train, X_test, y_test =         generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    clf_name = 'COPOD'
    clf = COPOD()
   
    X_train = pd.read_excel(r'__file__')
    
    X_train= np.array(X_train)
    
    X_test=X_train
    y_test=y_train
  
    clf.fit(X_train)
#     
    ts=clf.threshold_
    print('threshold',ts)
    
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_train)  # outlier labels (0 or 1)
    y_test_scores = -clf.decision_function(X_train)  # outlier scores

    
y_train_scores.sort()
n=len(y_train_scores)
pdf=y_train_scores
c=1
d1=[]
for i in range(len(pdf)-1):
    c=1
    sum=0
    sum+=pdf[i]
    b=(n-i)*pdf[i]
    c1=(sum+b)/(i+1)

    
    
    sum1=sum+pdf[i+1]
    b1=(n-i-1)*pdf[i+1]
    c2=(sum1+b1)/(i+2)
    
    
    d=c2/c1
    d1.append(d)
    print(d)
    

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
xmajorLocator  = MultipleLocator(20) 
xmajorFormatter = FormatStrFormatter('%1.1f') 
xminorLocator  = MultipleLocator(5) 
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(20,20))
c=pd.read_excel(r"index_y",index_col=None)
c = np.array(c)
c = c.reshape(c.shape[0]*c.shape[1], )
y=y_train_scores
x=c
plt.rcParams['font.family'] = 'Times New Roman' 
plt.xlabel("Sample number",fontsize=50)
plt.ylabel('Score',fontsize=50)

plt.scatter(x,y)
y6=[]
y7=[]
x6=[16,56,86,126,141,161,172,261,300,275]

for i in range(0,len(x6)):
    i2=x6[i]-1
    y7=y[i2]
    y6.append(y7)  
    
         

plt.scatter(x6,y6,s=550,color='m',marker=',',label="Tobacco stem sample")
y6=[]
y7=[]
x6=[6,191,234,252,289,314,39,76,213,101]

for i in range(0,len(x6)):
    i2=x6[i]-1
    y7=y[i2]
    y6.append(y7)  
    
         

plt.scatter(x6,y6,s=550,color='r',marker='>',label="Outlier due to physical factors")
for i in range(0,len(x)):
    if y[i]<6000:
         if i!=313:
            if i!=233:
                if i!=1:

                    x2=c[i]
                    y2=y[i]
                    plt.scatter(x2,y2,color='b',s=650,marker='*')
# for i in range(0,len(x)):
#     if y[i]<6000:
        
        
    
        
#         x2=x[i]
#         y2=y[i]
#         plt.scatter(x2,y2,color='b',s=650,marker='*')
# for i in range(0,len(x)):
#     if y[i]>6000:
#         if i!=3:
    
        
#             x3=x[i]
#             y3=y[i]
#             plt.scatter(x3,y3,color='r',s=550,marker='>')        
# for i in range(0,len(x)):
#     if y[i]>7028:
        
            
    
        
#                 x2=x[i]
#                 y2=y[i]
#                 plt.scatter(x2,y2,color='r',s=250,marker='>')                
# y6=[]
# y7=[]
  
i=1
plt.scatter(x[i],y[i],color='b',s=650,marker='*',label=u'Tobacco sample')   

plt.xlim(0,320)
import numpy as np
import matplotlib.pyplot as plt
#plt.axhline(y=7252, color='r', linestyle='-') 

# for i in range(0,len(x)):
#      if y[i]>6000:
#         plt.annotate(c[i],xy=(x[i],y[i]),xytext=(x[i],y[i]),fontsize=25)
plt.tick_params(labelsize=40)
plt.legend(fontsize=40,loc='upper left')
plt.savefig("fig.png")   
plt.show()    


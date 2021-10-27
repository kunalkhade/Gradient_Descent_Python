'''
    File name: test.py
    Supporting file: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020
    Date last modified: 10/30/2020
    Python Version: 3.7

    Topic 1: Develop generic binary classifier perceptron 
    class in ML.py.  It has to taketraining  set  of  any  size.   
    Class  must  include  four  functions  :init(),  fit()  ,netinput(), 
    predict(), One more supportive function to display result.


    Topic 2: Develop Linear Regression classifier using perceptron model 
    class Linear_Regression in ML.py.  It has to taketraining  set  of  any  size.   
    Class  must  include single function call Linear_Regression(A, B) and pass
    A = Learning Rate (0.01) and B = Iterations(5-10000). Call function final run and pass
    points [X, Y] array format. It will display result. 

    
'''
from ML import Perceptron
from ML import Linear_Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import ListedColormap

#pn instance variable pn assign to Perceptron class
#Pass learning rate = 0.25 and Iteration = 10
pn = Perceptron(0.1, 100)
lr = Linear_Regression(0.01, 50)
#Using Pandas import Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#Saperate parameters from dataset
x1 = df.iloc[0:50, 1].values
y1 = df.iloc[0:50, 0].values

x2 = df.iloc[51:100, 1].values
y2 = df.iloc[51:100, 0].values

x3 = df.iloc[101:150, 1].values
y3 = df.iloc[101:150, 0].values

#Make data compatible with function
points = np.stack((x1,y1),axis=1)
lr.final_run(points)

#Previous Programming assingment work
'''
#Only use initial 100 data value labels 
y = df.iloc[0:100, 4].values

#Convert labels into -1 and 1
y = np.where(y == 'Iris-setosa', -1, 1)

#Extract only 2 parameters from data set
X = df.iloc[0:100, [0, 2]].values

#Use fit, error, predict, weights, net_input functions from perceptron class
pn.fit(X, y)
print("Errors : \n", pn.error)
print("Prediction : \n",pn.predict(X)) 
print("Weights : \n", pn.weights)
#print(pn.net_input(X))

#Plot result 
pn.plot_decision_regions(X, y, classifier=pn, resolution=0.02)

#Plot Error function 
plt.plot(range(1, len(pn.error) + 1), pn.error,
marker='o')
plt.xlabel('Iteration')
plt.ylabel('# of misclassifications')
plt.show()

'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import radviz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set();
from sklearn.model_selection import cross_val_score
import math
from numpy.linalg import inv, solve, matrix_rank
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import sys, os
from random import shuffle, randint
from sklearn.metrics import confusion_matrix
from itertools import cycle
data=pd.read_csv('r/home/pegah/Desktop/Iris.csv')
data.head(5)
sns.heatmap(data.corr(), annot = True);
g = sns.pairplot(data,hue="Species")
data.describe()
data.Species.value_counts()
data.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
sns.FacetGrid(data, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
sns.boxplot(x="Species", y="PetalLengthCm", data=data)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=data)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=data, jitter=True, edgecolor="gray")
sns.FacetGrid(data, hue="Species", size=6) \
   .map(sns.kdeplot, "PetalLengthCm") \
   .add_legend()
sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3)
data.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

radviz(data.drop("Id", axis=1), "Species")
data.drop(['Id'],axis=1,inplace=True)
#The basis function(polynomial)
def poly_feats(input_values,degree):
  result=[]
  a=input_values[0]
  b=input_values[1]
  c=input_values[2]
  d=input_values[3]
  for i in range(degree):
    for j in range(degree):
      for k in range(degree):
        for l in range(degree):
          if i+j+k+l<=degree:
            result.append((a**i)*(b**j)*(c**k)*(d**l))
  for s in range(len(input_values)):
    result.append(input_values[s]**degree)
  return result

df=data.drop(['Id','SepalLengthCm','PetalLengthCm'],axis=1)
df['Species'].replace(to_replace=['Iris-setosa','Iris-virginica','Iris-versicolor'],value=[1.,-1.,-1.],inplace=True)
df.head()
np.random.seed(11)

# One-hot encoding of target label, Y
def one_hot(a):
  b = -1 * np.ones((a.size, a.max()+1))
  b[np.arange(a.size), a] = 1
  return b

# Loading digits datasets
iris = datasets.load_iris()

# One-hot encoding of target label, Y
Y = iris.target
#X = iris.data
#X_train,Y_train,X_test,Y_test=train_test_split(X, Y, shuffle=True, test_size = 0.2)
#y_trainp=one_hot(Y_train)
Y = one_hot(Y)

# Adding column of ones to absorb bias b of the hyperplane into X
X = iris.data
bias_ones = np.ones((len(X), 1))
X = np.hstack((X, bias_ones))

#bias_ones2 = np.ones((len(X_test), 1))
#X_testp = np.hstack((X_test, bias_ones2))
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, shuffle=True, test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

print("Training dataset: ", X_train.shape)
print("Validation dataset: ", X_val.shape)
print("Test dataset: ", X_test.shape)

def signum(x):
    if x >=0: return 1
    else : return -1

#General function for perceptron 
def Perceptron(X_train,Y_train,epochs,lr,result):
    result=[]
    epoch = 1
    m = 1
    # One vs Rest is a two class classification problem
    w = np.zeros((X_train.shape[1],1))     #Initialising the weights as 0
    while(m!=0 and epoch <= epochs):       # Iterating till epochs are reached or when error becomes 0
        m = 0 
        for xi,yi in zip(X_train,Y_train):  # Iterating over each sample and the corresponding class label
            y_hat = signum(np.dot(w.T,xi)[0]) 
                                            
            if yi*y_hat <0:                 #Condition for misclassified samples
                w = (w.T + yi*xi).T         ##Updating weights
                m = m + 1                   #Updating error count
        epoch = epoch + 1                   #Increasing epochs for looping
    return w,m

weights = np.zeros((X_train.shape[1],Y_train.shape[1])) # Defining a weight matrix (Num_of_features * Num of classes) to store each weight vectors
for i in range(Y_train.shape[1]):
    w,err = Perceptron(X_train,Y_train[:,i],100,1) #Getting the weight vector
    weights[:,i] = w[:,0]   
def predictclass(X_fit,Y_fit,weights,result):
    predictedclass = np.zeros(X_fit.shape[0])
    for i in range(X_fit.shape[0]):
        for j in range(Y_fit.shape[1]):
            predict = np.dot(weights[:,j],X_fit[i,:]) #Samplewise prediction
            if predict >0:
                predictedclass[i] = j
                break
    return predictedclass

def accuracy(Y_val,predictedclass):
    error = 0
    numsamples = Y_val.shape[0]
    for i in range(numsamples):
        Actualclass = Y_val[i,:]
        if Actualclass[int(predictedclass[i])]!=1.0:
             error+=1
    return (1-error/numsamples)
predictedclass = predictclass(X_val,Y_val,weights)
accuracy(Y_val,predictedclass) * 100
predictedclass = predictclass(X_test,Y_test,weights)
accuracy(Y_test,predictedclass) * 100
print(classification_report(Y_val, predictedclass))
print(confusion_matrix(Y_val,predictedclass))
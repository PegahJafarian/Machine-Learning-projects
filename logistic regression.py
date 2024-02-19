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

# approach for logistic regression

class LogisticRegression(object):
    
    def __init__(Logreg, alpha=0.01, n_iteration=100):  
        Logreg.alpha = alpha                            
        Logreg.n_iter = n_iteration
        
    def _sigmoid_function(Logreg, x): 
        value = 1 / (1 + np.exp(-x))
        return value
    def _cost_function(Logreg,h,theta, y): 
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost
    
    def _gradient_descent(Logreg,X,h,theta,y,m): 
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= Logreg.alpha * gradient_value
        return theta
    
    def fit(Logreg, X, y): 
        print("Fitting the given dataset..")
        Logreg.theta = []
        Logreg.cost = []
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y): 
            #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(Logreg.n_iter):
                z = X.dot(theta)
                h = Logreg._sigmoid_function(z)
                theta = Logreg._gradient_descent(X,h,theta,y_onevsall,m)
                cost.append(Logreg._cost_function(h,theta,y_onevsall)) 
            Logreg.theta.append((theta, i))
            Logreg.cost.append((cost,i))
        return Logreg

    def predict(Logreg, X): # this function calls the max predict function to classify the individul feauter
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [max((Logreg._sigmoid_function(i.dot(theta)), c) for theta, c in Logreg.theta)[1] for i in X ]

        return X_predicted

    def score(Logreg,X, y): #This function compares the predictd label with the actual label to find the model performance
        score = sum(Logreg.predict(X) == y) / len(y)
        return score
    
    def _plot_cost(Logreg,costh): # This function plot the Cost function value
        for cost,c in costh   :
                plt.plot(range(len(cost)),cost,'r')
                plt.title("Convergence Graph of Cost Function of type-" + str(c) +" vs All")
                plt.xlabel("Number of Iterations")
                plt.ylabel("Cost")
                plt.show()
                
# We are reading and processing the data provided
data = pd.read_csv('r/home/pegah/Desktop/Iris.csv')
#Transposing the data
data_T = data.T
data_T.dtypes
data_T.columns = ['Id','SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
                  

y_data = data_T['Species'].values  
X = data_T.drop(['Species','Id'],axis=1).values

scaler = StandardScaler()
X= scaler.fit_transform(X)
print(X)

for _ in range (10):
    X_train,X_test,y_train,y_test = train_test_split(X,y_data,test_size = 0.33)
    logi = LogisticRegression(n_iteration=30000).fit(X_train, y_train)
    predition1 = logi.predict(X_test)
    score1 = logi.score(X_test,y_test)
    print("the accuracy of the model is ",score1)
    scores.append(score1)
    
print(np.mean(scores))
print(classification_report(y_test, prediction1))
print(confusion_matrix(y_test, prediction1))
logi._plot_cost(logi.cost) # Here we ae plotting the Cost value and showing how it is depreciating close to 0 with each iteration

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

class_le = LabelEncoder()
y = class_le.fit_transform(data['Species'].values
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(data.iloc[:,range(0,4)].values)
S_W = np.zeros((4,4))
for i in range(3):
    S_W += np.cov(X_train_std[y==i].T)
print(S_W)

N=np.bincount(y) # number of samples for given class
vecs=[]
[vecs.append(np.mean(X_train_std[y==i],axis=0)) for i in range(3)] # class means
mean_overall = np.mean(X_train_std, axis=0) # overall mean
S_B=np.zeros((4,4))
for i in range(3):
    S_B += N[i]*(((vecs[i]-mean_overall).reshape(4,1)).dot(((vecs[i]-mean_overall).reshape(1,4))))
print(S_B)

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 5), discr, width=0.2,alpha=0.5, align='center',label='individual "discriminability"')
plt.step(range(1, 5), cum_discr, where='mid',label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()
W=np.hstack((eigen_pairs[0][1][:, ].reshape(4,1),eigen_pairs[1][1][:, ].reshape(4,1))).real
X_train_lda = X_train_std.dot(W)
data=pd.DataFrame(X_train_lda)
data['class']=y
data.columns=["LD1","LD2","class"]
data.head()
markers = ['s', 'x','o']
sns.lmplot(x="LD1", y="LD2", data=data, markers=markers,fit_reg=False, hue='class', legend=False)
plt.legend(loc='upper center')
plt.show()
from sklearn.metrics import classification_report
print(classification_report(X_train_lda, X_train_std))
#another approach for fisher
iris=datasets.load_iris()
X=iris.data
y=iris.target
def calcMC(X,Classes,result):
  mean=[]
  covariance=[]
  for i in range(Classes):
    u=np.mean(X[y==i],axis=0)
    mean.append(u)
    X_u=X-u
    sig=np.dot(X_u.T,X_u)
    covariance.append(sig)
  return mean,covariance
mean,covariance=calcMC(X,2)
print(mean)
print(covariance) 

def calcS(covariance):
  return np.sum(covariance,axis=0)
S=calcS(covariance)
print(S)  

def clacw(S,mean):
  S_inv=np.linalg.pinv(S)
  return np.dot(S_inv,mean[0]-mean[1])
w=clacw(S,mean)
print(w)

def predect(X,w,mean,result):
  shadow_c1=np.dot(w.T,mean[0])
  shadow_c2=np.dot(w.T,mean[1])
  y_pred=[]
  shadows=[]
  for i in range(len(X)):
    shadow=np.dot(w.T,X[i])
    shadows.append(shadow)
    if(shadow-shadow_c1)**2<(shadow-shadow_c2)**2:
      y=0
    else:
      y=1
    y_pred.append(y)
  return np.array(y_pred),np.array(shadows)

indexs=(y==0)|(y==1)
y_pred,shadows=predect(X[indexs],w,mean)

print(accuracy_score(y[indexs],y_pred))
print(precision_score(y[indexs], y_pred, average='micro'))
print(precision_recall_fscore_support(y[indexs], y_pred, average='macro'))

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

_, ax = plt.subplots(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)
lines, labels = [], []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")
plt.show()
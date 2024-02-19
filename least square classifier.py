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
#least square classification
  
def train(x,y,result):
	D = x.shape[1] + 1

	K = y.shape[1]
	
	sum1 = np.zeros((D,D)) 
	sum2 = np.zeros((D,K))
	i = 0
	for x_i in x:						
		x_i = np.append(1, x_i) 		 
		y_i = y[i]						
		sum1 += np.outer(x_i, x_i)		
		sum2 += np.outer(x_i, y_i)		
		i += 1
	
	while matrix_rank(sum1) != D:
		sum1 = sum1 + 0.001 * np.eye(D) 
	
	return np.dot(inv(sum1),sum2)


def predict(W, x,result):
	x = np.append(1, x)		
	values = list(np.dot(W.T,x))
	winners = [i for i, x in enumerate(values) if x == max(values)] 
	index = randint(0,len(winners)-1)
	winner = winners[index]

	y = [0 for x in values] 	
	y[winner] = 1 				
	return y

def fixLabels(y):
	newY = []
	for i in range(len(y)):
		size = max(y)
		temp = [0 for j in range(size + 1)]	
		temp[y[i]] = 1		
		newY.append(temp)	
	return np.matrix(newY)

def test(a,b, split):

	W = train(a[:split],b[:split])
	x = a[split:]
	y = b[split:]
	
	total = y.shape[0]
	i = 0
	hits = 0
	for i in range(total):
		prediction = predict(W,x[i])
		actual = list(y[i].A1)
		if prediction == actual:
			hits += 1
	accuracy = hits/float(total)*100
	print ("Accuracy = " + str(accuracy) + "%", "(" + str(hits) + "/" + str(total) + ")")

def usage():
	return 'usage: %s <data file> [head/tail]\n' % os.path.basename( sys.argv[ 0 ] )

def main():
	if len(sys.argv) < 2:
		print (usage())
		sys.exit(1)
	#head = False
	#if "--head" in sys.argv:
	#	head = True
	
	data = []
	classes = []
	f = open(sys.argv[1])
	try:
		
		for line in f:
			if line == "\n" or line == "": continue 
			line = line.strip("\n").split(",")		
			if head:
				
				data.append(map(lambda x: float(x), line[0:]))
				
				classes.append(line[0])
			else:
				
				data.append(map(lambda x: float(x), line[:-1]))
			
				classes.append(line[-1])
			
	finally:
		f.close()
	
	classes = map(lambda x: list(set(classes)).index(x), classes)
	x = np.matrix(data)
	y = fixLabels(classes)
	z = [] # temp array
	size = x.shape[0] - 1
	for i in range(size):
		z.append((x[i],y[i]))
	shuffle(z)
	for i in range(size):
		x[i] = z[i][0]
		y[i] = z[i][1]
		
	# scale data so it fits in range (0,1)
	for i in range(size):
		x[i] = x[i] / x.max()

	print( "20% Train/80% Test")
	split = int(size * 0.2)
	test(x,y,split)
	print ("\n")
    
if __name__ == "__main__":
	main()


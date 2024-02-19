import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import zscore
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
#pd.set_option('display.max_columns',None)
data = pd.read_csv(r'/home/pegah/Desktop/livingspace.csv')
data.head()
data.info()
round(data.isna().sum() * 100 / data.shape[0] , 2)#percentage for nall values
#delete columns with more than 50% null data
data.isna().sum()/len(data) 
data.isna() #where we have null values
data.columns[((data.isna().sum()/len(data))>0.50)]
data=data.drop(columns=data.columns[((data.isna().sum()/len(data))>0.50)]) #drop columns
data.columns
#drop meaningless data
data.drop(labels = ['scoutId'],axis = 'columns' , inplace=True)
#fillna numeric data 
#data._get_numeric_data().mean() 
data.fillna(data._get_numeric_data().mean(),inplace=True)
round(data.isna().sum() * 100 / data.shape[0] , 2)
data.isna().sum()
#normalize numeric values and delete outlier
for cols in data.columns:
    if data[cols].dtype=='int64' or data[cols].dtype=='float64':
        data[cols]=((data[cols]-data[cols].mean())/(data[cols].std()))
data.head() 
data.shape       
#delete outlier
#data.shape
for cols in data.columns:
    if data[cols].dtype=='int64' or data[cols].dtype=='float64':
        upper_range=data[cols].mean()+3*data[cols].std()
        lower_range=data[cols].mean()-3*data[cols].std()
        indexs=data[(data[cols]>upper_range)| (data[cols]<lower_range)].index
        data=data.drop(indexs)
data.shape
plt.figure(figsize=(28,8))
sns.countplot(data['heatingType'])
data.loc[:,'heatingType'].fillna('central_heating',inplace=True)

#fillna categorical data
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool':
        data[cols].fillna(data[cols].value_counts().head(1).index[0],inplace=True)
        #print('column: ' ,cols)
        #print(data[cols].value_counts().head(1).index[0])
        #print('cols:{},value:{}'.format(cols,data[cols].value_counts().head(1).index[0]))
        #print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))
#round(data.isna().sum() * 100 / data.shape[0] , 2)        
#categorical feature
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool':
        print('cols : {}, unique values : {}'.format(cols,data[cols].nunique()))
        
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool': 
        print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))

data.drop(labels = ['description','houseNumber','geo_bln','geo_krs','street','facilities','regio1','regio2','regio3','streetPlain'],axis = 'columns' , inplace=True)
others = list(data['heatingType'].value_counts().tail(12).index)
def edit_heatingType(x):
    if x in others:
        return 'other'
    else:
        return x

data['heatingType_edit'] = data['heatingType'].apply(edit_heatingType)
data = data.drop(columns = ['heatingType'])
data['heatingType_edit'].value_counts()*100 / len(data)

others = list(data['telekomTvOffer'].value_counts().tail(2).index)
def edit_telekomTvOffer(x):
    if x in others:
        return 'other'
    else:
        return x

data['telekomTvOffer_edit'] = data['telekomTvOffer'].apply(edit_telekomTvOffer)
data = data.drop(columns = ['telekomTvOffer'])
data['telekomTvOffer_edit'].value_counts()*100 / len(data)

others = list(data['firingTypes'].value_counts().tail(80).index)
def edit_firingTypes(x):
    if x in others:
        return 'other'
    else:
        return x

data['firingTypes_edit'] = data['firingTypes'].apply(edit_firingTypes)
data = data.drop(columns = ['firingTypes'])
data['firingTypes_edit'].value_counts()*100 / len(data)

others = list(data['condition'].value_counts().tail(7).index)
def edit_condition(x):
    if x in others:
        return 'other'
    else:
        return x

data['condition_edit'] = data['condition'].apply(edit_condition)
data = data.drop(columns = ['condition'])
data['condition_edit'].value_counts()*100 / len(data)

others = list(data['interiorQual'].value_counts().tail(3).index)
def edit_interiorQual(x):
    if x in others:
        return 'other'
    else:
        return x

data['interiorQual_edit'] = data['interiorQual'].apply(edit_interiorQual)
data = data.drop(columns = ['interiorQual'])
data['interiorQual_edit'].value_counts()*100 / len(data)

others = list(data['typeOfFlat'].value_counts().tail(9).index)
def edit_typeOfFlat(x):
    if x in others:
        return 'other'
    else:
        return x

data['typeOfFlat_edit'] = data['typeOfFlat'].apply(edit_typeOfFlat)
data = data.drop(columns = ['typeOfFlat'])
data['typeOfFlat_edit'].value_counts()*100 / len(data)

for cols in data.columns:
    if data[cols].dtype == 'object' or data[cols].dtype == 'bool':
        print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))

#categoricalcolumns
categoricalColumns = []
for cols in data.columns:
    if data[cols].dtype == 'object' or data[cols].dtype == 'bool':
        categoricalColumns.append(cols)
#categoricalcolumns  
dummies_feature = pd.get_dummies(data[categoricalColumns])
dummies_feature.head()
for item in dummies_feature.columns:
    if dummies_feature[item].dtype == 'bool':
      dummies_feature[item+'_edit'] = dummies_feature[item].astype(int)
dummies_feature.head()
data = pd.concat([data, dummies_feature], axis=1)
#data.head()
data = data.drop(columns=categoricalColumns)
data.head()
data.shape        
#correlation matrix
corr=data.corr()
f,ax=plt.subplots(figsize=(60,60))
sns.heatmap(corr,square=True , annot=True)
#plt.xticks(range(len(corr.columns)), corr.columns);
#plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

#Correlation with output variable
cor_target = corr["livingSpace"]
#Select highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
#Correlation with output variable
cor_target = corr["livingSpace"]
#Select highly correlated features
relevant_features = cor_target[cor_target< 0]
relevant_features
  
#linear regression
X = data[['totalRent', 'pricetrend', 'noRooms','floor','numberOfFloors','serviceCharge','baseRent']]
y = data['livingSpace']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#sum of Square Error
def cost_function(X, Y, B):
  J = np.sum((X.dot(B) - Y) ** 2)
  return J

def batch_gradient_descent(X, Y, B, alpha, iterations):
  cost_history = [0] * iterations
  p = len(Y)
  
  for iteration in range(iterations):
   
    h = X.dot(B)
    loss = h - Y
    gradient = X.T.dot(loss) / p
    B = B - alpha * gradient
    cost = cost_function(X, Y, B)
    cost_history[iteration] = cost
  
  return B, cost_history  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Initial Coefficients
B = np.zeros(X_train.shape[1])
alpha = 0.05
iter = 250
nB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter)
print(nB,cost_history)
plt.plot(cost_history)
plt.xlabel('Interations')
plt.ylabel('Training cost')
plt.show()
#approach for evaluate that the model is ok for test set
def pred(X,B):
  h = X.dot(B)
  return h
def RMSD_function(X,y):
    return np.sqrt(((X - y) ** 2).mean())
y_sample = pred(X_test,nB)
answer=RMSD_function(y_sample,y_test)
print(answer)
X_n = data[['totalRent', 'pricetrend', 'noRooms','floor','numberOfFloors','serviceCharge','baseRent']]
X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(X_n, y, test_size=0.2, random_state=0)
B = np.zeros(X_n_train.shape[1])
alpha = 0.05
iter = 250
nB_2, cost_history_2 = batch_gradient_descent(X_n_train, y_n_train, B, alpha, iter)
print(nB_2, cost_history_2)
plt.plot(cost_history_2)
plt.xlabel('Interations')
plt.ylabel('Training cost')
plt.show()
y_n_sample = pred(X_n_test,nB_2)
answer_n=RMSD_function(y_n_sample,y_n_test)
print(answer_n) 





        



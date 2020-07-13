import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
warnings.filterwarnings(action='ignore')

iris = pd.read_csv("Iris.csv",encoding='utf-8') #load acutal dataset

samples = [] # load bagging datasets
sample = pd.read_csv('Iris_bagging_dataset (1).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (2).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (3).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (4).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (5).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (6).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (7).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (8).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (9).csv', encoding = 'utf-8')
samples.append(sample)
sample = pd.read_csv('Iris_bagging_dataset (10).csv', encoding = 'utf-8')
samples.append(sample)

result = []
X = iris.iloc[:, 1:5] #extract predict features
y = iris.iloc[:, 5] #extract acutal result

for i in range(len(samples)): # 10 bagging rounds
    bagX = samples[i].iloc[:,1:5] #extract bagging's predict features
    bagy = samples[i].iloc[:,5] #extract bagging's actual result
    DTC = DecisionTreeClassifier(criterion='entropy') #make decision tree classifier
    DTC.fit(bagX,bagy) #fit with each bagging round's data
    predict = DTC.predict(X) #predict with each bagging round's fit classifier
    result.append(predict) #append each bagging round's result

rresult = []
for j in range(len(iris)): #majority voting
    a=0
    b=0
    c=0
    for k in range(len(samples)): #check which result is dominant
        if result[k][j]=='Iris-setosa':
            a+=1
        elif result[k][j]=='Iris-versicolor':
            b+=1
        else:
            c+=1
    if a>b and a>c: #append result with majority voting
        rresult.append('Iris-setosa')
    elif b>a and b>c:
        rresult.append('Iris-versicolor')
    else:
        rresult.append('Iris-virginica')

rresult = pd.DataFrame(rresult,columns=['Predict'])
confusion_matrix = pd.crosstab(y,rresult['Predict'],rownames=['Actual'],colnames=['Predicted']) #make confusion matrix with actual result and predicted result
print(confusion_matrix)
classification_report = classification_report(y,rresult['Predict']) # show accuracy with actual result and predicted result
print(classification_report)
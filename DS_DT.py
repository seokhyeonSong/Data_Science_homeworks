import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv('decision_tree_data.csv', encoding='utf-8') #read data from csv
df.head()
X = df[['level','lang','tweets','phd']] #derive data without target
y = df[['interview']] #derive target data
encoder = LabelEncoder() #encode categorical data into numerical data
encoder.fit(X[['level']])
df1 = encoder.transform(X[['level']])
encoder.fit(X[['lang']])
df2 = encoder.transform(X[['lang']])
encoder.fit(X[['tweets']])
df3 = encoder.transform(X[['tweets']])
encoder.fit(X[['phd']])
df4 = encoder.transform(X[['phd']])
data = {'level' : df1,
        'lang' : df2,
        'tweets' : df3,
        'phd':df4}
X = pd.DataFrame(data)
DTRegr = DecisionTreeClassifier(criterion='entropy',random_state=0)
scores = []
for i in range(1,10): #check highest accuracy from test_data percentage difference
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05*i)
        DTRegr.fit(X_train,y_train)
        scores.append(DTRegr.score(X_test,y_test))
print("with test dataset percentage ",(list(np.where(np.max(scores)==scores))[0]+1)*5,"shows highest acc")
print(np.max(scores)) #show highest accuracy
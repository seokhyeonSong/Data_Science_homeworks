import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv('linear_regression_data.csv', encoding='utf-8') #read csv dataset
df.head()
X = df[['Distance']] #derive data without target data
y = df[['Delivery Time']] #derive target data
linearRegr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #split dataset into 8:2=train_data:test_data
linearRegr.fit(X_train,y_train)
score = linearRegr.score(X_test,y_test) # show accuracy
print(score)
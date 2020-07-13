import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv('knn_data.csv', encoding='utf-8') #read data from csv
df['longitude'] = pd.to_numeric(df['longitude']) #data may be string but read them as numerical
df['latitude'] = pd.to_numeric(df['latitude'])
df.head()
X = df[['longitude','latitude']] #derive data without target
y = df[['lang']] #derive target data
scores = []
for i in range(1,24): #check all available n_neighbors
    knn = KNeighborsClassifier(n_neighbors=i)
    cv_scores = cross_val_score(knn,X,y,cv=5)
    scores.append(np.mean(cv_scores))
print(np.max(scores)) #show the highest accuracy
print("with n_neighbors ",list(np.where(np.max(scores)==scores))[0]+1)

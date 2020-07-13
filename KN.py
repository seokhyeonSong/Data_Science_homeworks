import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
"""
this is function for calculating euclidian distance between trained data and test data
parameter is trained data and test data
output is euclidian distance
"""
def euc(df,test):
    tmp = pow(df.iloc[0,0]-test.iloc[0,0],2)+pow(df.iloc[0,1]-test.iloc[0,1],2) # calculate euclidan distance
    return pow(tmp,0.5)

"""
this is function for normalize training data and test data
parameter is concatenated training data and test data
output is normalized training and test data
"""
def norm(df):
    scaler = StandardScaler() #normalize is used by standard scaler
    df = df.reset_index() #reindex concatenated data
    df = df.drop("index",axis=1)
    scaler.fit(df[['HEIGHT(cm)','WEIGHT(kg)']]) #normlize height and weight
    scaled = scaler.transform(df[['HEIGHT(cm)','WEIGHT(kg)']])
    scaled = pd.DataFrame(scaled,columns=[df.columns[0],df.columns[1]])
    scaled = pd.concat([scaled,df[[df.columns[2]]]],axis=1)
    return scaled
"""
this is function for predict the result by KNN
parameter is trained data, test data, neighbor value
output is list of prediction of test data
"""
def predict(df,test,k):
    result = [] # list for prediction
    for i in range(test.shape[0]):
        tmp = []
        ranking = [] # list for top k nearest index
        m = 0
        l = 0
        for j in range(df.shape[0]): # calculate euclidian distance and store them into a list
            a = euc(pd.DataFrame(df.loc[[j]]), pd.DataFrame(test.iloc[[i]]))
            tmp.append(a)
        for p in range(k):
            ranking.append(tmp.index(min(tmp))) # find nearest value
            tmp[tmp.index(min(tmp))] = 10000 # except nearest value because it is already contained to list
            if df.iloc[ranking[p], 2] == 'L': # count which target data is dominant
                l += 1
            else:
                m += 1
        if (l > m):
            result.append('L')
        else:
            result.append('M')
    return result
"""
this is function for overall KNN function
parameter is training data, test data and neighbor value
output is list of predction of test data
"""
def KN(df,test,k):
    scaled = norm(pd.concat([df,test]))
    test = scaled.iloc[df.shape[0]:scaled.shape[0]]
    df = scaled.iloc[0:df.shape[0]]
    return predict(df,test,k)
"""
this is function for get input value
there is no parameter value
output is dataframe type input value and numpy array type input value
"""
def val_input():
    count = (int)(input("how many datum to test : "))
    temp = np.empty([2,count])
    for i in range(count):
        temp[0][i] = input("Height in cm : ")
        temp[1][i] = input("Weight in kg : ")
    re_df = pd.DataFrame()
    for i in range(count):
        re_df = re_df.append(pd.DataFrame(data= [[(int)(temp[0][i]),(int)(temp[1][i])]], columns=["HEIGHT(cm)", "WEIGHT(kg)"]))
    return re_df,temp

df = pd.read_csv("KN.csv")
data,input_data = val_input()
result = (KN(df,data,(int)(input("K neighbor value : "))))
for i in range(len(result)):
    print("height : ",input_data[0][i],"\nweight : ",input_data[1][i])
    print("wears shirt size :",result[i],"\n")

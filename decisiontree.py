import pandas as pd
import math
import numpy as np
class Node(object): # class variable to store tree
    def __init__(self, attribute):
        self.center = None
        self.attr = attribute
        self.left = None
        self.right = None
        self.predict = None
"""
it it a function for calculating entropy of dataframe
parameter is dataframe and predict column name
output is entropy of dataframe
"""
def entropy(df,pre):
    df_p = df[df[pre]=="Respond"]
    df_n = df[df[pre]=="Nothing"]
    p = df_p.shape[0]
    n = df_n.shape[0]
    if p == 0 or n == 0 : #if there is only Respond or Nothing
        return 0
    else :
        return -((p/(p+n))*math.log2(p/(p+n))+(n/(p+n))*math.log2(n/(p+n)))
"""
it is a function for calculating information gain of dataframe
parameter is dataframe and predict column name
output is the column name of highest information gain value's column
"""
def ingain(df,pre):
    rootent = entropy(df, pre) # calculate root entropy
    ent = np.array([])
    for i in range(len(df.columns)-1): # consider all of the columns without predict column
        tmp = 0
        for j in range(len(df.iloc[:, i].unique())): # consider every distinct value of columns
            tmpdf = df[df[df.columns[i]] == df.iloc[:, i].unique()[j]]
            if tmpdf.shape[0] > 1 : # if there is no value in specific case, it causes error while calculate log, though ignore this case, it anyway converges to 0 which doesn't affect the result
                tmp += entropy(tmpdf,pre) * tmpdf.shape[0]
        ent = np.append(ent,np.array(rootent - tmp/df.shape[0])) #collect all information gain value
    return df.columns[np.argmax(ent)] # return the highest information gain's column name
"""
it is a function for build decision tree
parameter is dataframe and predict column name
output is the tree object class
"""
def decisiontree(df,pre):
    if entropy(df,pre) == 0 : #if there is only Nothing or Respond which is tree's leaf
        tree = Node(None)
        if df[df[pre]=="Respond"].shape[0]>df[df[pre]=="Nothing"].shape[0]: # check whether nothing or respond
            tree.predict = "Respond"
        else:
            tree.predict = "Nothing"
        return tree
    else:
        root = ingain(df,pre) #select root attriute of tree
        tree = Node(root) #make tree with selected attribute
        for i in range(len(df[root].unique())): # consider all kinds of values
            if df[root].unique()[i] == "No" or df[root].unique()[i] =="Suburban" or df[root].unique()[i] == "Detached" or df[root].unique()[i] == "High":
                #in case of some values, they go to left
                tree.left = decisiontree(df[df[root]==df[root].unique()[i]].drop(root,axis=1), pre)
            elif df[root].unique()[i] == "Yes" or df[root].unique()[i] == "Rural" or df[root].unique()[i] == "Semi-detached" or df[root].unique()[i] =="Low":
                #in case of some other values, tehy go to center
                tree.center = decisiontree(df[df[root] == df[root].unique()[i]].drop(root, axis=1), pre)
            elif df[root].unique()[i] == "Urban":
                #in case of Urban of district which only has 3 kind of values, it goes to right
                tree.right = decisiontree(df[df[root] == df[root].unique()[i]].drop(root, axis=1), pre)
        return tree
"""
it is a function for predict result with given data
parameter is given data and made tree
output is the result whether customer would respond or not
"""
def predict(df,tree):
    if(tree.predict == "Nothing") or (tree.predict == "Respond"): #if it is leaf node of tree
        return tree.predict
    elif df[tree.attr][0] == "No": # case of go to left node
        result = predict(df,tree.left)
        return result
    elif df[tree.attr][0] == "Yes": # case of go to center node
        result = predict(df,tree.center)
        return result
    elif df[tree.attr][0] == "Suburban": # case of go to left node
        result = predict(df,tree.left)
        return result
    elif df[tree.attr][0] == "Rural": # case of go to center node
        result = predict(df,tree.center)
        return result
    elif df[tree.attr][0] == "Urban":# case of go to right node
        result = predict(df,tree.right)
        return result
    elif df[tree.attr][0] == "Detached": # case of go to left node
        result = predict(df,tree.left)
        return result
    elif df[tree.attr][0] == "Semi-detached": # case of go to center node
        result = predict(df,tree.center)
        return result
    elif df[tree.attr][0] == "High": # case of go to left node
        result = predict(df,tree.left)
        return result
    elif df[tree.attr][0] == "Low": # case of go to center node
        result = predict(df,tree.center)
        return result
"""
It is function for receive test dataset
There is no parameter
output is dataframe type value which is made by given value 
"""
def val_input():
    data = np.empty([4, 1])
    print("select number")
    data[0] = (int)(input("1. urban 2. suburban 3. rural\nDistrict : "))
    data[1] = (int)(input("1. detatched 2. semi-detached\nHouse type : "))
    data[2] = (int)(input("1. low 2. high\nIncome : "))
    data[3] = (int)(input("1. no 2. yes\nPrevious customer : "))
    datastr = []
    if (data[0] == 1):
        datastr.append("Urban")
    elif (data[0] == 2):
        datastr.append("Suburban")
    else:
        datastr.append("Rural")
    if (data[1] == 1):
        datastr.append("Detached")
    else:
        datastr.append("Semi-detached")
    if (data[2] == 1):
        datastr.append("Low")
    else:
        datastr.append("High")
    if (data[3] == 1):
        datastr.append("No")
    else:
        datastr.append("Yes")
    test = pd.DataFrame({"District": [datastr[0]], "House Type": [datastr[1]], "Income": [datastr[2]],
                             "Previous Customer": [datastr[3]]})
    return test

df = pd.read_csv("decisionTree.csv") # load training data
tree = decisiontree(df, "Outcome")  # build tree with given data
test = val_input()
#given data to predict result
print("prediction : " , predict(test,tree))


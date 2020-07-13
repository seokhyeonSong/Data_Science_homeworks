import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model

data = pd.read_csv("bmi_data_lab3.csv")
origindata = data

print("number of sex is female",len(data[data['Sex']=='Female'])) # number of sex is female
print("number of sex is male", len(data[data['Sex']=='Male'])) # number of sex is male
print("number of under 30 ages", len(data[data['Age']<30])) # number of under 30 ages
print("number of over 30 ages", len(data[data['Age']>=30])) # number of over 30 ages
print("mean age", data['Age'].mean()) # mean age
print("number of under 67 inches", len(data[data['Height (Inches)']<67])) # number of under 67 inches
print("number of over 67 inches", len(data[data['Height (Inches)']>67])) # number of over 67 inches
print("mean height", data['Height (Inches)'].mean()) # mean height
print("number of under 130 pounds", len(data[data['Weight (Pounds)']<130])) # number of under 130 pounds
print("number of over 130 pounds", len(data[data['Weight (Pounds)']>130])) # number of over 130 pounds
print("mean weight ", data['Weight (Pounds)'].mean()) # mean weight
print("number of BMI 0", len(data[data['BMI']==0])) # number of BMI 0
print("number of BMI 1", len(data[data['BMI']==1])) # number of BMI 1
print("number of BMI 2", len(data[data['BMI']==2]))# number of BMI 2
print("number of BMI 3", len(data[data['BMI']==3]))# number of BMI 3
print("number of BMI 4", len(data[data['BMI']==4]))# number of BMI 4
print("mean BMI", data['BMI'].mean()) # mean BMI
print("data column names", data.columns) # data column names
print("BMI 3's height \n", data[data['BMI']==3].iloc[1:len(data),2]) # BMI 3's height
fig = plt.figure() # figure variable
ax1 = fig.add_subplot(431) # divide place
ax1.title.set_text("BMI=1 / heights") # give title for subplot
plt.hist(x=np.array(data[data['BMI']==1]['Height (Inches)']),bins=10,range=(62,75)) # make histogram
ax2 = fig.add_subplot(433)
ax2.title.set_text("BMI=2 / heights")
plt.hist(x=np.array(data[data['BMI']==2]['Height (Inches)']),bins=10,range=(62,75))
ax3 = fig.add_subplot(435)
ax3.title.set_text("BMI=3 / heights")
plt.hist(x=np.array(data[data['BMI']==3]['Height (Inches)']),bins=10,range=(62,75))
ax4 = fig.add_subplot(437)
ax4.title.set_text("BMI=1 / weights")
plt.hist(x=np.array(data[data['BMI']==1]['Weight (Pounds)']),bins=10,range=(90,160))
ax5 = fig.add_subplot(439)
ax5.title.set_text("BMI=2 / weights")
plt.hist(x=np.array(data[data['BMI']==2]['Weight (Pounds)']),bins=10,range=(90,140))
ax6 = fig.add_subplot(4,3,11)
ax6.title.set_text("BMI=3 / weights")
plt.hist(x=np.array(data[data['BMI']==3]['Weight (Pounds)']),bins=10,range=(110,160))
fig.suptitle("plot histogram for BMI value") # entire title
plt.show()
SScaler = sklearn.preprocessing.StandardScaler() # Standardscaler variable
MScaler = sklearn.preprocessing.MinMaxScaler() # MinMaxScaler variable
RScaler = sklearn.preprocessing.RobustScaler() # RobustScaler variable
SScaled = SScaler.fit_transform(np.array(data.drop('Sex',axis=1))) # fit and scale with data
MScaled = MScaler.fit_transform(np.array(data.drop('Sex',axis=1)))
RScaled = RScaler.fit_transform(np.array(data.drop('Sex',axis=1)))
SScaled = pd.DataFrame(SScaled,columns=['Age','Height (Inches)','Weight (Pounds)','BMI']) # make scaled data as dataframe
MScaled = pd.DataFrame(MScaled,columns=['Age','Height (Inches)','Weight (Pounds)','BMI'])
RScaled = pd.DataFrame(RScaled,columns=['Age','Height (Inches)','Weight (Pounds)','BMI'])
fig = plt.figure() # another subplot variable
ax1 = fig.add_subplot(441) # divide place
ax1.title.set_text("before scale / height") # set subplot's title
plt.hist(x=np.array(data['Height (Inches)']),bins=10) # make histogram
ax1 = fig.add_subplot(443)
ax1.title.set_text("Standardscaler / height")
plt.hist(x=pd.DataFrame(SScaled['Height (Inches)']),bins=10)
ax1 = fig.add_subplot(446)
ax1.title.set_text("Minmaxscaler / height")
plt.hist(x=pd.DataFrame(MScaled['Height (Inches)']),bins=10)
ax1 = fig.add_subplot(448)
ax1.title.set_text("RobustScaler / height")
plt.hist(x=pd.DataFrame(RScaled['Height (Inches)']),bins=10)
ax1 = fig.add_subplot(449)
ax1.title.set_text("before scale / weight")
plt.hist(x=np.array(data['Weight (Pounds)']),bins=10)
ax1 = fig.add_subplot(4,4,11)
ax1.title.set_text("Standardscaler / weight")
plt.hist(x=pd.DataFrame(SScaled['Weight (Pounds)']),bins=10)
ax1 = fig.add_subplot(4,4,14)
ax1.title.set_text("Minmaxscaler / weight")
plt.hist(x=pd.DataFrame(MScaled['Weight (Pounds)']),bins=10)
ax1 = fig.add_subplot(4,4,16)
ax1.title.set_text("RobustScaler / weight")
plt.hist(x=pd.DataFrame(RScaled['Weight (Pounds)']),bins=10)
fig.suptitle("plot height & weight for each scalar") # entire title
plt.show() # draw plot

dropindex = [] # space for drop index number
i=0
for i in range(len(data)-1):
    if not(((float)(data.iloc[i,1])> 0) and ((float)(data.iloc[i,1]) < 100)) : # remove likely - wrong values with age
        data.iloc[i,1] = np.nan
    if not(((float)(data.iloc[i,2])>30) and ((float)(data.iloc[i,2])<90)) : # remove likely - wrong values with height
        data.iloc[i,2] = np.nan
    if not(((float)(data.iloc[i,3])>50) and ((float)(data.iloc[i,3])<150)): # remove likely - wrong values with weight
        data.iloc[i,3] = np.nan
    if not(((float)(data.iloc[i,4])>=0) or ((float)(data.iloc[i,4])<5)): # remove likely - wrong values with BMI
        dropindex.append(i)
    elif not(data.iloc[i,0]=="Male" or data.iloc[i,0]=="Female"): # remove likely - wrong values with sex
        dropindex.append(i)

manipuldata = data.drop(data.index[dropindex]) # drop rows with BMI not equals 0,1,2,3,4 or sex not equals Male,Female

print("numbers of rows with nan : ",len(manipuldata)-len(manipuldata.dropna(how='any'))) # numbers of rows with nan
print("numbers of columns(sex) with nan : ",len(manipuldata['Sex'])-len(manipuldata['Sex'].dropna(how='any'))) # numbers of columns(sex) with nan
print("numbers of columns(age) with nan : ", len(manipuldata['Age'])-len(manipuldata['Age'].dropna(how='any'))) # numbers of columns(age) with nan
print("numbers of columns(height) with nan : ",len(manipuldata['Height (Inches)'])-len(manipuldata['Height (Inches)'].dropna(how='any'))) # numbers of columns(height) with nan
print("numbers of columns(weight) with nan : ",len(manipuldata['Weight (Pounds)'])-len(manipuldata['Weight (Pounds)'].dropna(how='any'))) # numbers of columns(weight) with nan
print("numbers of columns(BMI) with nan : ", len(manipuldata['BMI'])-len(manipuldata['BMI'].dropna(how='any'))) # numbers of columns(bmi) with nan
print("extract all rows without NAN\n",manipuldata.dropna(how='any')) # extract all rows without NAN
print("fill nan with ffill", manipuldata.fillna(method='ffill')) # fill nan with ffill
print("fill nan with bfill", manipuldata.fillna(method='bfill')) # fill nan with bfill
meandata = manipuldata
mediandata = manipuldata
meandata['Age'] = (manipuldata['Age'].fillna(manipuldata['Age'].mean()))
meandata['Height (Inches)'] = (manipuldata['Height (Inches)'].fillna(manipuldata['Height (Inches)'].mean()))
meandata['Weight (Pounds)'] = (manipuldata['Weight (Pounds)'].fillna(manipuldata['Weight (Pounds)'].mean()))
meandata['BMI'] = (manipuldata['BMI'].fillna(manipuldata['BMI'].mean()))
print("fill nan with mean")
print(meandata) # fill nan with mean
mediandata = manipuldata
mediandata['Age'] = (manipuldata['Age'].fillna(manipuldata['Age'].median()))
mediandata['Height (Inches)'] = (manipuldata['Height (Inches)'].fillna(manipuldata['Height (Inches)'].median()))
mediandata['Weight (Pounds)'] = (manipuldata['Weight (Pounds)'].fillna(manipuldata['Weight (Pounds)'].median()))
mediandata['BMI'] = (manipuldata['BMI'].fillna(manipuldata['BMI'].median()))
print("fill nan with median")
print(mediandata) # fill nan with median

manipuldata = data
weight = np.array(manipuldata.dropna(how='any')['Weight (Pounds)']) # extract weight values which don't have nan
height = np.array(manipuldata.dropna(how='any')['Height (Inches)']) # extract height values which don't have nan
nandata = np.array(manipuldata.loc[manipuldata.isnull()['Height (Inches)']]['Weight (Pounds)']) # extract dirty values

# linear regression to predict height
reg = sklearn.linear_model.LinearRegression() # linear regression variable
reg.fit(weight[:, np.newaxis], height) # fit model with given dirty value
px = np.array([weight.min()-1, weight.max()+1]) # set min & max x dot to draw equation
py = reg.predict(px[:, np.newaxis]) # set min & max y dot which predicted by before px to draw equation
plt.scatter(weight, height, c='b') # dirty value contained histogram
plt.scatter(nandata,reg.predict(nandata.reshape(-1,1)),c='g') # cleaned dirty value histogram
plt.plot(px,py,color='r') # draw linear regression equation
plt.xlabel("weight") # x label
plt.ylabel("height") # y label
plt.title("all data") # title
plt.show() # draw

# linear regression grouped by sex(man) to predict height
weight_man = np.array(manipuldata[manipuldata['Sex']=='Male'].dropna(how='any')['Weight (Pounds)']) # extract man's weight values which don't have nan
height_man = np.array(manipuldata[manipuldata['Sex']=='Male'].dropna(how='any')['Height (Inches)']) # extract man's height values which don't have nan
nandata_man = np.array(manipuldata[manipuldata['Sex']=='Male'].loc[manipuldata.isnull()['Height (Inches)']]['Weight (Pounds)']) # extract man's dirty weight values

reg_man = sklearn.linear_model.LinearRegression()
reg_man.fit(weight_man[:, np.newaxis], height_man)
px_man = np.array([weight_man.min()-1, weight_man.max()+1])
py_man = reg_man.predict(px_man[:, np.newaxis])
plt.scatter(weight_man, height_man, c='b')
plt.scatter(nandata_man,reg_man.predict(nandata_man.reshape(-1,1)),c='g')
plt.plot(px_man,py_man,color='r')
plt.xlabel("weight")
plt.ylabel("height")
plt.title("man data")
plt.show()

# linear regression grouped by sex(woman) to predict height
weight_woman = np.array(manipuldata[manipuldata['Sex']=='Female'].dropna(how='any')['Weight (Pounds)']) # extract woman's weight values which don't have nan
height_woman = np.array(manipuldata[manipuldata['Sex']=='Female'].dropna(how='any')['Height (Inches)']) # extract woman's height values which don't have nan
nandata_woman = np.array(manipuldata[manipuldata['Sex']=='Female'].loc[manipuldata.isnull()['Height (Inches)']]['Weight (Pounds)'])# extract woman's dirty weight values

reg_woman = sklearn.linear_model.LinearRegression()
reg_woman.fit(weight_woman[:, np.newaxis], height_woman)
px_woman = np.array([weight_woman.min()-1, weight_woman.max()+1])
py_woman = reg_woman.predict(px_woman[:, np.newaxis])
plt.scatter(weight_woman, height_woman, c='b')
plt.scatter(nandata_woman,reg_woman.predict(nandata_woman.reshape(-1,1)),c='g')
plt.plot(px_woman,py_woman,color='r')
plt.xlabel("weight")
plt.ylabel("height")
plt.title("woman data")
plt.show()

# linear regression grouped by BMI(2) to predict height
weight_BMI2 = np.array(manipuldata[manipuldata['BMI']==2].dropna(how='any')['Weight (Pounds)']) # extract BMI 2's weight values which don't have nan
height_BMI2 = np.array(manipuldata[manipuldata['BMI']==2].dropna(how='any')['Height (Inches)']) # extract BMI 2's height values which don't have nan
nandata_BMI2 = np.array(manipuldata[manipuldata['BMI']==2].loc[manipuldata.isnull()['Height (Inches)']]['Weight (Pounds)'])# extract BMI 2's dirty weight values

reg_BMI2 = sklearn.linear_model.LinearRegression()
reg_BMI2.fit(weight_BMI2[:, np.newaxis], height_BMI2)
px_BMI2 = np.array([weight_BMI2.min()-1, weight_BMI2.max()+1])
py_BMI2 = reg_BMI2.predict(px_BMI2[:, np.newaxis])
plt.scatter(weight_BMI2, height_BMI2, c='b')
plt.scatter(nandata_BMI2,reg_BMI2.predict(nandata_BMI2.reshape(-1,1)),c='g')
plt.plot(px_BMI2,py_BMI2,color='r')
plt.xlabel("weight")
plt.ylabel("height")
plt.title("BMI 2 data")
plt.show()

# linear regression grouped by BMI(3) to predict height
weight_BMI3 = np.array(manipuldata[manipuldata['BMI']==3].dropna(how='any')['Weight (Pounds)']) # extract weight values which don't have nan
height_BMI3 = np.array(manipuldata[manipuldata['BMI']==3].dropna(how='any')['Height (Inches)']) # extract height values which don't have nan
nandata_BMI3 = np.array(manipuldata[manipuldata['BMI']==3].loc[manipuldata.isnull()['Height (Inches)']]['Weight (Pounds)'])# extract BMI 3's dirty weight values

reg_BMI3= sklearn.linear_model.LinearRegression()
reg_BMI3.fit(weight_BMI3[:, np.newaxis], height_BMI3)
px_BMI3 = np.array([weight_BMI3.min()-1, weight_BMI3.max()+1])
py_BMI3 = reg_BMI3.predict(px_BMI3[:, np.newaxis])
plt.scatter(weight_BMI3, height_BMI3, c='b')
plt.scatter(nandata_BMI3,reg_BMI3.predict(nandata_BMI3.reshape(-1,1)),c='g')
plt.plot(px_BMI3,py_BMI3,color='r')
plt.xlabel("weight")
plt.ylabel("height")
plt.title("BMI 3 data")
plt.show()

# multiple equations with variable groupes to predict height
plt.plot(px,py,color='y',label='all')
plt.plot(px_man,py_man,color='k',label='man')
plt.plot(px_woman,py_woman,color='m',label='woman')
plt.plot(px_BMI2,py_BMI2,color='r',label='BMI2')
plt.plot(px_BMI3,py_BMI3,color='g',label='BMI3')
plt.xlabel("weight")
plt.ylabel("height")
plt.title("compare several version")
plt.legend()
plt.show()

nandata = np.array(manipuldata.loc[manipuldata.isnull()['Weight (Pounds)']]['Height (Inches)'])# extract dirty height values
nandata_man = np.array(manipuldata.loc[manipuldata.isnull()['Weight (Pounds)']]['Height (Inches)'])# extract man's dirty height values
nandata_woman = np.array(manipuldata.loc[manipuldata.isnull()['Weight (Pounds)']]['Height (Inches)'])# extract woman's dirty height values
nandata_BMI2 = np.array(manipuldata.loc[manipuldata.isnull()['Weight (Pounds)']]['Height (Inches)'])# extract BMI 2's dirty height values
nandata_BMI3 = np.array(manipuldata.loc[manipuldata.isnull()['Weight (Pounds)']]['Height (Inches)'])# extract BMI 3's dirty height values
# linear regression to predict weight
reg = sklearn.linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min()-1, height.max()+1])
py = reg.predict(px[:, np.newaxis])
plt.scatter(height, weight, c='b')
plt.scatter(nandata,reg.predict(nandata.reshape(-1,1)),c='g')
plt.plot(px,py,color='r')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("all data")
plt.show()

# linear regression grouped by sex(man) to predict weight
reg_man = sklearn.linear_model.LinearRegression()
reg_man.fit(height_man[:, np.newaxis], weight_man)
px_man = np.array([height_man.min()-1, height_man.max()+1])
py_man = reg_man.predict(px_man[:, np.newaxis])
plt.scatter(height_man, weight_man, c='b')
plt.scatter(nandata_man,reg_man.predict(nandata_man.reshape(-1,1)),c='g')
plt.plot(px_man,py_man,color='r')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("man data")
plt.show()

# linear regression grouped by sex(woman) to predict weight
reg_woman = sklearn.linear_model.LinearRegression()
reg_woman.fit(height_woman[:, np.newaxis], weight_woman)
px_woman = np.array([height_woman.min()-1, height_woman.max()+1])
py_woman = reg_woman.predict(px_woman[:, np.newaxis])
plt.scatter(height_woman, weight_woman, c='b')
plt.scatter(nandata_woman,reg_woman.predict(nandata_woman.reshape(-1,1)),c='g')
plt.plot(px_woman,py_woman,color='r')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("woman data")
plt.show()

# linear regression grouped by BMI(2) to predict weight
reg_BMI2 = sklearn.linear_model.LinearRegression()
reg_BMI2.fit(height_BMI2[:, np.newaxis], weight_BMI2)
px_BMI2 = np.array([height_BMI2.min()-1, height_BMI2.max()+1])
py_BMI2 = reg_BMI2.predict(px_BMI2[:, np.newaxis])
plt.scatter(height_BMI2, weight_BMI2, c='b')
plt.scatter(nandata_BMI2,reg_BMI2.predict(nandata_BMI2.reshape(-1,1)),c='g')
plt.plot(px_BMI2,py_BMI2,color='r')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("BMI 2 data")
plt.show()

# linear regression grouped by BMI(3) to predict weight
reg_BMI3= sklearn.linear_model.LinearRegression()
reg_BMI3.fit(height_BMI3[:, np.newaxis], weight_BMI3)
px_BMI3 = np.array([height_BMI3.min()-1, height_BMI3.max()+1])
py_BMI3 = reg_BMI3.predict(px_BMI3[:, np.newaxis])
plt.scatter(height_BMI3, weight_BMI3, c='b')
plt.scatter(nandata_BMI3,reg_BMI3.predict(nandata_BMI3.reshape(-1,1)),c='g')
plt.plot(px_BMI3,py_BMI3,color='r')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("BMI 3 data")
plt.show()

# multiple equations with variable groupes to predict weight
plt.plot(px,py,color='y',label='all')
plt.plot(px_man,py_man,color='k',label='man')
plt.plot(px_woman,py_woman,color='m',label='woman')
plt.plot(px_BMI2,py_BMI2,color='r',label='BMI2')
plt.plot(px_BMI3,py_BMI3,color='g',label='BMI3')
plt.xlabel("height")
plt.ylabel("weight")
plt.title("compare several version")
plt.legend()
plt.show()
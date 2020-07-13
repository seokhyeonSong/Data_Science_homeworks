import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import math

data = pd.read_excel("bmi_data_phw3.xlsx")

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
ax1 = fig.add_subplot(451) # divide place
ax1.title.set_text("BMI=0 / heights") # give title for subplot
plt.hist(x=np.array(data[data['BMI']==0]['Height (Inches)']),bins=10) # make histogram
ax2 = fig.add_subplot(453)
ax1.title.set_text("BMI=1 / heights")
plt.hist(x=np.array(data[data['BMI']==1]['Height (Inches)']),bins=10) # make histogram
ax3 = fig.add_subplot(455)
ax3.title.set_text("BMI=2 / heights")
plt.hist(x=np.array(data[data['BMI']==2]['Height (Inches)']),bins=10)
ax4 = fig.add_subplot(457)
ax4.title.set_text("BMI=3 / heights")
plt.hist(x=np.array(data[data['BMI']==3]['Height (Inches)']),bins=10)
ax5 = fig.add_subplot(459)
ax5.title.set_text("BMI=3 / heights")
plt.hist(x=np.array(data[data['BMI']==4]['Height (Inches)']),bins=10)
ax6 = fig.add_subplot(4,5,11)
ax6.title.set_text("BMI=0 / weights")
plt.hist(x=np.array(data[data['BMI']==0]['Weight (Pounds)']),bins=10)
ax7 = fig.add_subplot(4,5,13)
ax7.title.set_text("BMI=1 / weights")
plt.hist(x=np.array(data[data['BMI']==1]['Weight (Pounds)']),bins=10)
ax8 = fig.add_subplot(4,5,15)
ax8.title.set_text("BMI=2 / weights")
plt.hist(x=np.array(data[data['BMI']==2]['Weight (Pounds)']),bins=10)
ax9 = fig.add_subplot(4,5,17)
ax9.title.set_text("BMI=3 / weights")
plt.hist(x=np.array(data[data['BMI']==3]['Weight (Pounds)']),bins=10)
ax10 = fig.add_subplot(4,5,19)
ax10.title.set_text("BMI=4 / weights")
plt.hist(x=np.array(data[data['BMI']==4]['Weight (Pounds)']),bins=10)
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

manipuldata = data
weight = np.array(manipuldata.dropna(how='any')['Weight (Pounds)']) # extract weight values which don't have nan
height = np.array(manipuldata.dropna(how='any')['Height (Inches)']) # extract height values which don't have nan

# linear regression to predict height
reg = sklearn.linear_model.LinearRegression() # linear regression variable
reg.fit(height[:, np.newaxis], weight) # fit model with given dirty value
LRweight = reg.predict(height.reshape(-1,1)) # cleaned dirty value histogram
Zscore = (LRweight-np.mean(LRweight))/np.std(LRweight) # get z-score
plt.hist(Zscore,bins=10) # make histogram with z-score
plt.xlabel("Ze")
plt.ylabel("frequency")
plt.title("All sex e value")
plt.show()
Zscore.sort()
a = math.fabs(Zscore[math.ceil((len(Zscore)*0.04))]) \
    +math.fabs(Zscore[math.ceil((len(Zscore)*0.9))]) # according to korean staticks there are 4% underweight, 10% overweight so i set a with low 4% and high 10%
a=a/2 #average
print("my alpha value to choose BMI 0 & 4 : ", a)
count0 = 0
count4 = 0
i=0
for i in range(len(Zscore)) : # get estimate BMI 0 & 4
    if Zscore[i] < -a :
        count0=count0+1
    if Zscore[i] > a :
        count4=count4+1
real0 = len(manipuldata[manipuldata['BMI']==0]) # real BMI 0 count
real4 = len(manipuldata[manipuldata['BMI']==4]) # real BMI 4 count
plt.bar(np.arange(4),[count0,real0,count4,real4],align='center') # draw bar chart
plt.xticks(np.arange(4),['estimate 0','real 0','estimate 4', 'real4'])
plt.title("real data vs estimate data / all sex")
plt.show()


Df = data[data['Sex']=='Female'] #for female data
weight = np.array(Df.dropna(how='any')['Weight (Pounds)']) # extract weight values which don't have nan
height = np.array(Df.dropna(how='any')['Height (Inches)']) # extract height values which don't have nan

regf = sklearn.linear_model.LinearRegression() # linear regression variable
regf.fit(height[:, np.newaxis], weight) # fit model with given dirty value
LRweightf = regf.predict(height.reshape(-1,1)) # cleaned dirty value histogram
Zscoref = (LRweightf-np.mean(LRweightf))/np.std(LRweightf)
plt.hist(Zscoref,bins=10)
plt.xlabel("Ze")
plt.ylabel("frequency")
plt.title("Female e value")
plt.show()
Zscoref.sort()
af = math.fabs(Zscoref[math.floor((len(Zscoref)*0.04))])+math.fabs(Zscoref[math.floor((len(Zscoref)*0.9))])
af=af/2
print("my alpha value to choose BMI 0 & 4 among female: ", af)
count0 = 0
count4 = 0
i=0
for i in range(len(Zscore)) :
    if Zscore[i] < -af :
        count0=count0+1
    if Zscore[i] > af :
        count4=count4+1
real0 = len(Df[Df['BMI']==0])
real4 = len(Df[Df['BMI']==4])
plt.bar(np.arange(4),[count0,real0,count4,real4],align='center')
plt.xticks(np.arange(4),['estimate 0','real 0','estimate 4', 'real4'])
plt.title("real data vs estimate data / Female")
plt.show()


Dm = data[data['Sex']=="Male"] # for male data
weight = np.array(Dm.dropna(how='any')['Weight (Pounds)']) # extract weight values which don't have nan
height = np.array(Dm.dropna(how='any')['Height (Inches)']) # extract height values which don't have nan

regm = sklearn.linear_model.LinearRegression() # linear regression variable
regm.fit(height[:, np.newaxis], weight) # fit model with given dirty value
LRweightm = regm.predict(height.reshape(-1,1)) # cleaned dirty value histogram
Zscorem = (LRweightm-np.mean(LRweightm))/np.std(LRweightm)
plt.hist(Zscorem,bins=10)
plt.xlabel("Ze")
plt.ylabel("frequency")
plt.title("Male e value")
plt.show()
Zscorem.sort()
am = math.fabs(Zscorem[math.floor((len(Zscorem)*0.04))])+math.fabs(Zscorem[math.floor((len(Zscorem)*0.9))])
am=am/2
count0 = 0
count4 = 0
i=0
for i in range(len(Zscore)) :
    if Zscore[i] < -am :
        count0=count0+1
    if Zscore[i] > am :
        count4=count4+1
real0 = len(Dm[Dm['BMI']==0])
real4 = len(Dm[Dm['BMI']==4])
plt.bar(np.arange(4),[count0,real0,count4,real4],align='center')
plt.xticks(np.arange(4),['estimate 0','real 0','estimate 4', 'real4'])
plt.title("real data vs estimate data / Male")
plt.show()

print("my alpha value to choose BMI 0 & 4 : among male", am)
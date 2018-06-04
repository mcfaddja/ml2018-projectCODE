import readline
import scipy.sparse
import scipy
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
import xml.etree.ElementTree as ET
from sklearn.externals import joblib
import time
import sys


likes = pd.read_csv("/Users/jamster/data/training/relation/relation.csv")

likesUIDs = likes.ix[:,1].values
likesLIDs = likes.ix[:,2].values
lsLikesUIDs = likesUIDs.tolist()
lsLikesLIDs = likesLIDs.tolist()
setLikesUIDs = set(lsLikesUIDs)
setLikesLIDs = set(lsLikesLIDs)
unqLikesUIDs = (list(setLikesUIDs))
unqLikesLIDs = (list(setLikesLIDs))
allLikesLS = [lsLikesUIDs, [str(x) for x in lsLikesLIDs]]
allLikesLS = list(map(list, zip(*allLikesLS)))

aDictLikes2 = {}
for aUID in unqLikesUIDs:
	aDictLikes2[aUID]=[]

for row in allLikesLS:
	aDictLikes2[row[0]].append(row[1])

combDICT = {}
for uid in unqLikesUIDs:
	tmpDICT={}
	tmpLS = aDictLikes2[uid]
	for row in tmpLS:
		tmpDICT[str(row)]=1
	combDICT[uid]=tmpDICT

tryTHIS=[]
for uid in unqLikesUIDs:
	tryTHIS.append(combDICT[uid])


v = DictVectorizer()
likesMAT=v.fit_transform(tryTHIS)


del globals()['likes']
del globals()['likesUIDs']
del globals()['likesLIDs']
del globals()['lsLikesUIDs']
del globals()['lsLikesLIDs']
del globals()['setLikesUIDs']
del globals()['setLikesLIDs']
del globals()['allLikesLS']
del globals()['aDictLikes2']
del globals()['aUID']
del globals()['row']
del globals()['combDICT']
del globals()['uid']
del globals()['tmpDICT']
del globals()['tmpLS']
del globals()['tryTHIS']
del globals()['v']



profilesDF=pd.read_csv("/Users/jamster/data/training/profile/profile.csv")
profiles=profilesDF.ix[:,1:9].values.copy()
profilesLSo=profiles.tolist().copy()

profilesLS=[]
for row in profilesLSo:
	tmpLS=row
	tmpAGE=row[1]

	if tmpAGE < 25:
		tmpLS[1]=1
	elif tmpAGE < 35:
		tmpLS[1]=2
	elif tmpAGE < 50:
		tmpLS[1]=3
	else:
		tmpLS[1]=4

	profilesLS.append(tmpLS)


profsTOlikes=[]
for i in range(len(profilesLS)):
	profsTOlikes.append([])

for row in profilesLS:
	tmpIND = unqLikesUIDs.index(row[0])
	profsTOlikes[tmpIND]=row


profsTOlikes1=list(map(list, zip(*profsTOlikes)))

sexsARR=np.array(profsTOlikes1[2])


del globals()['unqLikesUIDs']
del globals()['unqLikesLIDs']
del globals()['profilesDF']
del globals()['profiles']
del globals()['profilesLSo']
del globals()['profilesLS']
del globals()['row']
del globals()['tmpLS']
del globals()['tmpAGE']
del globals()['profsTOlikes']
del globals()['i']
del globals()['tmpIND']



seed = 7
myRand = np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(likesMAT, sexsARR, test_size=1500)
numInputs = int(likesMAT.shape[1])


# myInputs = Input(shape=(numInputs,))

# x = Sequential()


# model = Sequential()
# model.add(Dense(int(numInputs*1.5),
# 				input_dim=numInputs,
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense((numInputs*2),
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dropout(0.375))
# model.add(Dense(int(numInputs*1.5),
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(numInputs,
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# model.compile(optimizer='adam',
# 				loss='binary_crossentropy',
# 				metrics=['accuracy', 'mse'])


# model = Sequential()
# model.add(Dense(int(numInputs*1.5),
# 				input_dim=numInputs,
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(numInputs,
# 				kernel_initializer='uniform',
# 				activation='relu'))
# model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# model.compile(optimizer='adam',
# 				loss='binary_crossentropy',
# 				metrics=['accuracy', 'mse'])


numInputs = int(sys.argv[1])

model = Sequential()
model.add(Dense(int(numInputs*1.5),
				input_dim=int(likesMAT.shape[1]),
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dropout(0.25))
model.add(Dense((numInputs*2),
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(int(numInputs*1.5),
				kernel_initializer='uniform',
				activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(numInputs,
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

model.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', 'mse'])

model.fit(X_train, y_train, epochs=10)

y_pred = model.predict(X_test)
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))
print("sexs, keras: ", str(numInputs), " ", model.evaluate(X_test, y_test))
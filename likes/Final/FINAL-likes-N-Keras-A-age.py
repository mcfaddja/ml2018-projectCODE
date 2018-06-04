import readline
import scipy.sparse
import scipy
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
from keras.models import model_from_json
from sklearn.externals import joblib
import sys
import os



# Import the 'LIKES' from the training dataset and process
###############
likes = pd.read_csv("/Users/jamster/data/training/relation/relation.csv")

# Extact individual columns and convert to lists
likesUIDs = likes.ix[:,1].values
likesLIDs = likes.ix[:,2].values
lsLikesUIDs = likesUIDs.tolist()
lsLikesLIDs = likesLIDs.tolist()

# Convert columns to sets
setLikesUIDs = set(lsLikesUIDs)
setLikesLIDs = set(lsLikesLIDs)

# Convert columns to list of unique items
unqLikesUIDs = (list(setLikesUIDs))
unqLikesLIDs = (list(setLikesLIDs))

# Get list of all User IDs (UIDs) paried with the Like IDs (LIDs) of the 
#   posts the user has liked
allLikesLS = [lsLikesUIDs, [str(x) for x in lsLikesLIDs]]
allLikesLS = list(map(list, zip(*allLikesLS)))


# Convert list of UID and LID pairs into a dictionary indexed by UIDs
aDictLikes2 = {}
for aUID in unqLikesUIDs:
	aDictLikes2[aUID]=[]

for row in allLikesLS:
	aDictLikes2[row[0]].append(row[1])


# Convert into a dictionary (by UIDs) of dictionaries (by LIDs)
combDICT = {}
for uid in unqLikesUIDs:
	tmpDICT={}
	tmpLS = aDictLikes2[uid]
	for row in tmpLS:
		tmpDICT[str(row)]=1
	combDICT[uid]=tmpDICT


# Convert 'combDICT' into a list of dictionaries (of LIDs)
tryTHIS=[]
for uid in unqLikesUIDs:
	tryTHIS.append(combDICT[uid])

# Vectorize the list of dictionaries in 'tryTHIS' to get the UID/LID matrix 
#   for the training data
v = DictVectorizer()
likesMAT=v.fit_transform(tryTHIS)



# Clear unused variable to free memory
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



# Import the profiles from the training dataset
###############
profilesDF=pd.read_csv("/Users/jamster/data/training/profile/profile.csv")

# Get the values of the relevant columns and convert them to a list
profiles=profilesDF.ix[:,1:9].values.copy()
profilesLSo=profiles.tolist().copy()

# Categorize the ages
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


# Align the profiles data with the indexing of the likes data
profsTOlikes=[]
for i in range(len(profilesLS)):
	profsTOlikes.append([])

for row in profilesLS:
	tmpIND = unqLikesUIDs.index(row[0])
	profsTOlikes[tmpIND]=row

profsTOlikes1=list(map(list, zip(*profsTOlikes)))

# Extract Data for AGEs
agesARRo=np.array(profsTOlikes1[1])
agesARRo=agesARRo.tolist()

# Convert data for AGEs to binary vectors
agesARR = []
for row in agesARRo:
	if row==1:
		agesARR.append([1,0,0,0])
	elif row==2:
		agesARR.append([0,1,0,0])
	elif row==3:
		agesARR.append([0,0,1,0])
	else:
		agesARR.append([0,0,0,1])

agesARR=np.array(agesARR)



# Clear MORE unused variable to free memory
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



# Training Model
###############
print("start training")

numInputs = 750 # Number of nodes to map inputs

model = Sequential()
model.add(Dense(int(numInputs*1.5),
				input_dim=int(likesMAT.shape[1]),
				activation='relu'))
model.add(Dropout(0.25))
model.add(Dense((numInputs*2),
				activation='softmax'))
model.add(Dropout(0.375))
model.add(Dense(int(numInputs*1.5),
				activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(numInputs,
				activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy', 'mse'])

model.fit(likesMAT, agesARR, epochs=10)

print("end training")



# Save model
###############
print("start writing JSON file")
# serialize model to JSON
model_json = model.to_json()
with open("AGES-Keras.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
print("start writing H5 file")
model.save_weights("AGES-Keras.h5")

print("Saved AGES model from Keras to disk")
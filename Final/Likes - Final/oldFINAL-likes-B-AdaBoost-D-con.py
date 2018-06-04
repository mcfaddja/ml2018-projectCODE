import readline
import scipy.sparse
import scipy
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib
import sys



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

# Extract data for CONs
consARR=np.array(profsTOlikes1[4])



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

nEST = 500
lR = 1.0
adaBoost = AdaBoostRegressor(n_estimators=nEST, learning_rate=lR)
adaBoost.fit(likesMAT, consARR)

print("training complete")



# Save model
###############
joblib.dump(adaBoost, "/Users/jamster/adaBoost-A-cons.xz", compress=9)
print("model saved to disk")



print("DONE")
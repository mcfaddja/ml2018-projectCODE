import random
import numpy as np
import pandas as pd
import os
import sys
from sklearn.feature_extraction import DictVectorizer
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



pathLIKEs = "/data/training/relation"
pathPROFs = "/data/training/profile"
test_pathLIKEs = sys.argv[1]+"/relation"
test_pathPROFs = sys.argv[1]+"/profile"
# test_pathLIKEs = "/data/public-test-data/relation"
# test_pathPROFs = "/data/public-test-data/profile"

dfLIKES = pd.read_csv(pathLIKEs + "/relation.csv")
likesUIDs = dfLIKES.ix[:,1].values
likesLIDs = dfLIKES.ix[:,2].values
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
print("done with processing likes matrix from training data")




# print(len(unqLikesUIDs))
# print(len(unqLikesLIDs))
# print(type(unqLikesUIDs))
# print(type(unqLikesLIDs))




test_dfLIKES = pd.read_csv(test_pathLIKEs + "/relation.csv")
# print(test_dfLIKES.shape)
# test_likesUIDs = test_dfLIKES.ix[:,1].values
# test_likesLIDs = test_dfLIKES.ix[:,2].values
# test_lsLikesUIDs = test_likesUIDs.tolist()
# test_lsLikesLIDs = test_likesLIDs.tolist()
# test_setLikesUIDs = set(test_lsLikesUIDs)
# test_setLikesLIDs = set(test_lsLikesLIDs)
# test_unqLikesUIDs = (list(test_setLikesUIDs))
# test_unqLikesLIDs = (list(test_setLikesLIDs))
# test_allLikesLS = [test_lsLikesUIDs, [str(x) for x in test_lsLikesLIDs]]
# test_allLikesLS = list(map(list, zip(*test_allLikesLS)))

# print(len(test_unqLikesUIDs))


test_dfLIKES = test_dfLIKES[ test_dfLIKES['like_id'].isin(unqLikesLIDs)]
# print(test_dfLIKES.shape)

test_likesUIDs = test_dfLIKES.ix[:,1].values
test_likesLIDs = test_dfLIKES.ix[:,2].values
test_lsLikesUIDs = test_likesUIDs.tolist()
test_lsLikesLIDs = test_likesLIDs.tolist()
test_setLikesUIDs = set(test_lsLikesUIDs)
test_setLikesLIDs = set(test_lsLikesLIDs)
test_unqLikesUIDs = (list(test_setLikesUIDs))
test_unqLikesLIDs = (list(test_setLikesLIDs))
test_allLikesLS = [test_lsLikesUIDs, [str(x) for x in test_lsLikesLIDs]]
test_allLikesLS = list(map(list, zip(*test_allLikesLS)))

# print(len(test_unqLikesUIDs))

# print(unqLikesLIDs[1])
# print(test_lsLikesLIDs.index(unqLikesLIDs[1]))
# print(test_lsLikesLIDs[test_lsLikesLIDs.index(unqLikesLIDs[1])])

# print(len(test_unqLikesUIDs))

test_aDictLikes2 = {}
for aUID in test_unqLikesUIDs:
	test_aDictLikes2[aUID]=[]

# print(test_aDictLikes2)


for row in test_allLikesLS:
	test_aDictLikes2[row[0]].append(row[1])

# test_aDictLikes2["dummy"] = unqLikesLIDs


# print(test_aDictLikes2)


test_combDICT = {}
for uid in test_unqLikesUIDs:
	tmpDICT={}
	tmpLS = test_aDictLikes2[uid]
	for row in tmpLS:
		tmpDICT[str(row)]=1
	test_combDICT[uid]=tmpDICT
    

test_tryTHIS=[]
for uid in test_unqLikesUIDs:
	test_tryTHIS.append(test_combDICT[uid])

# print(test_combDICT[uid])
# print(type(test_combDICT[uid]))
# print(type(unqLikesLIDs))

dummyDICT = {}
for row in unqLikesLIDs:
    dummyDICT[str(row)] = 1

test_tryTHIS.append(dummyDICT)
test_unqLikesUIDs.append("dummy")
# print(len(test_unqLikesUIDs))
# print(test_unqLikesUIDs)

test_v = DictVectorizer()
test_likesMAT=test_v.fit_transform(test_tryTHIS)
print("done with processing likes matrix from test data")

# print(test_likesMAT.shape)
# print(test_likesMAT[317,:])
# print(test_likesMAT[318,:])

# test_likesMATo = np.delete(test_likesMAT, 318, 0)
# test_likesMAT.row_del(318)
# print(test_likesMAT[317,:])
# print(test_likesMAT[318,:])
# print(test_likesMATo[317,:])
# print(test_likesMATo[318,:])


dfPROFS = pd.read_csv(pathPROFs + "/profile.csv")
profiles = dfPROFS.ix[:,1:9].values.copy()
profilesLSo = profiles.tolist().copy()

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

print("finished processing profile matrix from training data")



test_dfPROFS = pd.read_csv(test_pathPROFs + "/profile.csv")
test_profiles = test_dfPROFS.ix[:,1].values.copy()
test_profilesLSo = test_profiles.tolist().copy()

test_profilesLS=[]
for row in test_profilesLSo:
	tmpLS=row
	test_profilesLS.append(tmpLS)

test_profilesLS.append("dummy")
# print(len(test_profilesLS))

# print(test_profilesLS)

test_profsTOlikes=[]
for i in range(len(test_unqLikesUIDs)):
	test_profsTOlikes.append([])

# print(len(test_profsTOlikes))
# print(test_profsTOlikes)

# print(test_unqLikesUIDs)

notINlist = []
for row in test_profilesLS:
    if row in test_unqLikesUIDs:
	    tmpIND = test_unqLikesUIDs.index(row)
    else:
        notINlist.append(row)
    test_profsTOlikes[tmpIND]=row

# print(len(test_profsTOlikes))
# print(test_profsTOlikes)
# print(len(notINlist))

test_profsTOlikes1=list(map(list, zip(*test_profsTOlikes)))

print("finished processing profile matrix from test data")

# print(notINlist)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
from keras.models import model_from_json

# json_file = open("/Users/jamster/mcfaddja-models/AGES-Keras.json", 'r')
json_file = open("mcfaddja-models/AGES-Keras.json", 'r')
agesMODEL_json = json_file.read()
json_file.close()
agesMODEL = model_from_json(agesMODEL_json)
# agesMODEL.load_weights("/Users/jamster/mcfaddja-models/AGES-Keras.h5")
agesMODEL.load_weights("mcfaddja-models/AGES-Keras.h5")
print("loaded AGES model from disk")

agesMODEL.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy', 'mse'])

pred_ages = agesMODEL.predict(test_likesMAT)
print("AGES model has made predictions")

# print(pred_ages)
# print(pred_ages[1])
# print(type(pred_ages))
# print(type(pred_ages[1]))

# i = 0
# for row in pred_ages:
#     tmpROW = np.around(row, decimals=0)
#     pred_ages[i]=tmpROW
#     i+=1

proc_ages = []
for row in pred_ages:
    tmpGRP = np.argmax(row)
    tmpGRP+=1
    proc_ages.append(tmpGRP)

# print(proc_ages)
print("predictions by AGES model have been processed")



# json_file = open("/Users/jamster/mcfaddja-models/SEXS-Keras.json", 'r')
json_file = open("mcfaddja-models/SEXS-Keras.json", 'r')
sexsMODEL_json = json_file.read()
json_file.close()
sexsMODEL = model_from_json(sexsMODEL_json)
# sexsMODEL.load_weights("/Users/jamster/mcfaddja-models/SEXS-Keras.h5")
sexsMODEL.load_weights("mcfaddja-models/SEXS-Keras.h5")
print("loaded SEXS model from disk")

sexsMODEL.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', 'mse'])

pred_sexs = sexsMODEL.predict(test_likesMAT)
print("SEXS model has made predictions")

# print(pred_sexs)
# print(pred_sexs[1])
# print(type(pred_sexs))
# print(type(pred_sexs[1]))

proc_sexs = []
for row in pred_sexs:
    tmpSEX = np.around(row, decimals=0)
    proc_sexs.append(int(tmpSEX))

# print(proc_sexs)
print("predictions by SEXS model have been processed")




from sklearn.externals import joblib

# opesMODEL = joblib.load("/Users/jamster/mcfaddja-models/SVM-A-opes.xz")
opesMODEL = joblib.load("mcfaddja-models/SVM-A-opes.xz")
print("loaded OPES model from disk")

pred_opes = opesMODEL.predict(test_likesMAT)
# print(pred_opes)



print("yay!")


def create_xmlA(i,user):
    root = ET.Element("user")
    if proc_ages[i]==1:
	    root.set("age_group","xx-24")
    elif proc_ages[i]==2:
        root.set("age_group","25-34")
    elif proc_ages[i]==3:
        root.set("age_group","35-49")
    else:
        root.set("age_group","50-xx")

    if proc_sexs[i]==0:
        root.set("gender","male")
    else:
        root.set("gender","female")

    root.set("id", test_profsTOlikes[i])

    root.set("extrovert","3.49")

    root.set("neurotic","2.73")

    root.set("agreeable","3.58")

    root.set("conscientious","3.45")

    root.set("open", str(np.around(pred_opes[i], decimals=2)))
    tree = ET.ElementTree(root)
    tree.write(sys.argv[2]+"/"+str(user)+".xml")
    # tree.write("/Users/jamster/output/"+str(user)+".xml")

    # return 0



def create_xmlB(i,user):
    root = ET.Element("user")
    root.set("age_group","xx-24")
    root.set("gender","female")
    root.set("id", notINlist[i])
    root.set("extrovert","3.49")
    root.set("neurotic","2.73")
    root.set("agreeable","3.58")
    root.set("conscientious","3.45")
    root.set("open","3.91")
    tree = ET.ElementTree(root)
    tree.write(sys.argv[2]+"/"+str(user)+".xml")
    # tree.write("/Users/jamster/output/"+str(user)+".xml")


# i=0
# print(test_profilesLS)
# for user in test_profilesLS:
#     ind = test_profilesLS.index(user)
#     print(ind)
#     create_xmlA(ind, user)
#     # i+=1
#     # print(i)

# print(notINlist)
for user in notINlist:
    ind  = notINlist.index(user)
    create_xmlB(ind, user)

for user in test_profsTOlikes:
    if user != "dummy":
        ind = test_profsTOlikes.index(user)
        create_xmlA(ind, user)




# # NB->age and gender.
# df = pd.read_csv("/data/training/profile/profile.csv")
# df['text']=''
# for i in range(0,len(df)):
# 	#print(row['userid']+info.get(row['userid']))
#     print(df.loc[i,'userid']+info.get(df['userid']))
# 	df.loc[i,'text'] = info.get(df.loc[i,'userid'])
# 	age = int(df.loc[i,'age'])
# 	if(age<=24):
# 		df.loc[i,'age']="xx-24"
# 	elif(age>=25 and age<=34):
# 		df.loc[i,'age']="25-34"
# 	elif(age>=35 and age<=49):
# 		df.loc[i,'age']="35-49"
# 	else:
# 		df.loc[i,'age']="50-xx"
# course: TCSS555
# ML in Python, Term Project
# date: 06/01/2018
# name: Jonathan McFadden
# description: Code to run the models of MyFacebook data which were generated 
#               from the 'LIKES' data


# Import initial required libraries
###############
import random
import numpy as np
import pandas as pd
import os
import sys
from sklearn.feature_extraction import DictVectorizer
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




# Set file/directory paths
###############
pathLIKEs = "/data/training/relation"
pathPROFs = "/data/training/profile"
# test_pathLIKEs = sys.argv[1]+"/relation"
# test_pathPROFs = sys.argv[1]+"/profile"
test_pathLIKEs = "/data/public-test-data/relation"
test_pathPROFs = "/data/public-test-data/profile"




# Import the 'LIKES' from the training dataset and process
###############
dfLIKES = pd.read_csv(pathLIKEs + "/relation.csv")

# Extact individual columns and convert to lists
likesUIDs = dfLIKES.ix[:,1].values
likesLIDs = dfLIKES.ix[:,2].values
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
lsOfDicts=[]
for uid in unqLikesUIDs:
    lsOfDicts.append(combDICT[uid])

# Vectorize the list of dictionaries in 'lsOfDicts' to get the UID/LID matrix 
#   for the training data
v = DictVectorizer()
likesMAT=v.fit_transform(lsOfDicts)

print("done with processing likes matrix from training data")




# Import the 'LIKES' from the test dataset and process
###############
test_dfLIKES = pd.read_csv(test_pathLIKEs + "/relation.csv")

# remove any data (by like_id) which does not occur in the training data
test_dfLIKES = test_dfLIKES[ test_dfLIKES['like_id'].isin(unqLikesLIDs)]

# Extact individual columns and convert to lists
test_likesUIDs = test_dfLIKES.ix[:,1].values
test_likesLIDs = test_dfLIKES.ix[:,2].values
test_lsLikesUIDs = test_likesUIDs.tolist()
test_lsLikesLIDs = test_likesLIDs.tolist()

# Convert columns to sets
test_setLikesUIDs = set(test_lsLikesUIDs)
test_setLikesLIDs = set(test_lsLikesLIDs)

# Convert columns to list of unique items
test_unqLikesUIDs = (list(test_setLikesUIDs))
test_unqLikesLIDs = (list(test_setLikesLIDs))

# Get list of all User IDs (UIDs) paried with the Like IDs (LIDs) of the 
#   posts the user has liked
test_allLikesLS = [test_lsLikesUIDs, [str(x) for x in test_lsLikesLIDs]]
test_allLikesLS = list(map(list, zip(*test_allLikesLS)))

# test_dfLIKES.to_csv("/Users/jamster/Desktop/test.csv")


# Convert list of UID and LID pairs into a dictionary indexed by UIDs
test_aDictLikes2 = {}
for aUID in test_unqLikesUIDs:
    test_aDictLikes2[aUID]=[]

for row in test_allLikesLS:
    test_aDictLikes2[row[0]].append(row[1])

# test_aDictLikes2["dummy"] = unqLikesLIDs


# Convert into a dictionary (by UIDs) of dictionaries (by LIDs)
test_combDICT = {}
for uid in test_unqLikesUIDs:
    tmpDICT={}
    tmpLS = test_aDictLikes2[uid]
    for row in tmpLS:
        tmpDICT[str(row)]=1
    test_combDICT[uid]=tmpDICT
    
# Convert 'test_combDICT' into a list of dictionaries (of LIDs)
test_lsOfDicts=[]
for uid in test_unqLikesUIDs:
    test_lsOfDicts.append(test_combDICT[uid])


# Create a dictionary of dummy LID values to cover ALL LIDs in the training 
#   dataset, this way the test and training like matrices have the same 
#   number of columns
dummyDICT = {}
for row in unqLikesLIDs:
    dummyDICT[str(row)] = 1

# Append the dummy dictionary to the list of dictionaries and append the 
#   UID 'dummy' to the list of UIDs
test_lsOfDicts.append(dummyDICT)
test_unqLikesUIDs.append("dummy")


# Vectorize the list of dictionaries in 'test_lsOfDicts' to get the UID/LID matrix 
#   for the test data
test_v = DictVectorizer()
test_likesMAT = test_v.fit_transform(test_lsOfDicts)

# tmp0 = test_likesMAT * np.ones((test_likesMAT.shape[1], 1))

print("done with processing likes matrix from test data")




# Import the profiles from the training dataset
###############
dfPROFS = pd.read_csv(pathPROFs + "/profile.csv")

# Get the values of the relevant columns and convert them to a list
profiles = dfPROFS.ix[:,1:9].values.copy()
profilesLSo = profiles.tolist().copy()

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

print("finished processing profile matrix from training data")




# Import the profiles from the test dataset
###############
test_dfPROFS = pd.read_csv(test_pathPROFs + "/profile.csv")

# Get the values of the relevant columns and convert them to a list
test_profiles = test_dfPROFS.ix[:,1].values.copy()
test_profilesLSo = test_profiles.tolist().copy()

# Prep the profile data from the test dataset in the same manner as the 
#   profile data from the training dataset
test_profilesLS=[]
for row in test_profilesLSo:
    tmpLS=row
    test_profilesLS.append(tmpLS)

test_profilesLS.append("dummy") # Append the 'dummy' User ID described eariler


# Align the profiles data with the indexing of the likes data and generate 
#   a list of User IDs with no likes information in the test dataset
test_profsTOlikes=[]
for i in range(len(test_unqLikesUIDs)):
    test_profsTOlikes.append([])

notINlist = []
for row in test_profilesLS:
    if row in test_unqLikesUIDs:
        tmpIND = test_unqLikesUIDs.index(row)
    else:
        notINlist.append(row)
    test_profsTOlikes[tmpIND]=row

test_profsTOlikes1=list(map(list, zip(*test_profsTOlikes)))

print("finished processing profile matrix from test data")




# Import Keras Libraries
###############
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
from keras.models import model_from_json



# Import Keras AGES model from file
###############
json_file = open("/home/itadmin/mcfaddja-models/AGES-Keras.json", 'r')
# json_file = open("/Users/jamster/mcfaddja-models/AGES-Keras.json", 'r')
agesMODEL_json = json_file.read()
json_file.close()
agesMODEL = model_from_json(agesMODEL_json)
agesMODEL.load_weights("/home/itadmin/mcfaddja-models/AGES-Keras.h5")
# agesMODEL.load_weights("/Users/jamster/mcfaddja-models/AGES-Keras.h5")

print("loaded AGES model from disk")


# Complile Keras AGES model and predict
###############
agesMODEL.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy', 'mse'])

pred_ages = agesMODEL.predict(test_likesMAT)

print("AGES model has made predictions")


# Process the predictions of the Keras AGES model
###############
proc_ages = []
for row in pred_ages:
    tmpGRP = np.argmax(row)
    tmpGRP+=1
    proc_ages.append(tmpGRP)

print("predictions by AGES model have been processed")




# Import Keras SEXS model from file
###############
json_file = open("/home/itadmin/mcfaddja-models/SEXS-Keras.json", 'r')
# json_file = open("/Users/jamster/mcfaddja-models/SEXS-Keras.json", 'r')
sexsMODEL_json = json_file.read()
json_file.close()
sexsMODEL = model_from_json(sexsMODEL_json)
sexsMODEL.load_weights("/home/itadmin/mcfaddja-models/SEXS-Keras.h5")
# sexsMODEL.load_weights("/Users/jamster/mcfaddja-models/SEXS-Keras.h5")

print("loaded SEXS model from disk")

# Complile Keras SEXS model and predict
###############
sexsMODEL.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', 'mse'])

pred_sexs = sexsMODEL.predict(test_likesMAT)
print("SEXS model has made predictions")


# Process the predictions of the Keras SEXS model
###############
proc_sexs = []
for row in pred_sexs:
    tmpSEX = np.around(row, decimals=0)
    proc_sexs.append(int(tmpSEX))

# print(proc_sexs)
print("predictions by SEXS model have been processed")




# Import sklearn libraries required for importing saved models
###############
# from sklearn.externals import joblib



# Import SKLearn Support Vector Machine OPEs model from file and predict
###############
# # opesMODEL = joblib.load("/Users/jamster/mcfaddja-models/SVM-A-opes.xz")
# opesMODEL = joblib.load("mcfaddja-models/SVM-A-opes.xz")
# print("loaded OPES model from disk")

# pred_opes = opesMODEL.predict(test_likesMAT)
# # print(pred_opes)
# print("OPES model has made predictions")




# Import SKLearn Bagging (w/ in-bag-scoring) CONs model from file and predict
###############
# consMODEL = joblib.load("mcfaddja-models/bagIN-A-cons.xz")
# print("loaded CONS model from disk")

# pred_cons = consMODEL.predict(test_likesMAT)
# print("CONS model has made predictions")




# Import SKLearn Bagging (w/ in-bag-scoring) EXTs model from file and predict
###############
# extsMODEL = joblib.load("mcfaddja-models/bagIN-A-cons.xz")
# print("loaded EXTS model from disk")

# pred_exts = extsMODEL.predict(test_likesMAT)
# print("EXTS model has made predictions")




# Import SKLearn Bagging (w/ in-bag-scoring) AGRs model from file and predict
###############
# agrsMODEL = joblib.load("mcfaddja-models/bagIN-A-agrs.xz")
# print("loaded AGRS model from disk")

# pred_agrs = agrsMODEL.predict(test_likesMAT)
# print("AGRS model has made predictions")




# Import SKLearn K-Nearest Neighbors NEUs model from file and predict
###############
# neusMODEL = joblib.load("mcfaddja-models/knn-A-neus.xz")
# print("loaded NEUS model from disk")

# pred_neus = neusMODEL.predict(test_likesMAT)
# print("NEUS model has made predictions")




# Create XML files for users with LIKES data in the test dataset
###############
def create_xmlA(i,user):
    root = ET.Element("user")

    # AGES
    if proc_ages[i]==1:
	    root.set("age_group","xx-24")
    elif proc_ages[i]==2:
        root.set("age_group","25-34")
    elif proc_ages[i]==3:
        root.set("age_group","35-49")
    else:
        root.set("age_group","50-xx")

    # GENDER/SEXS
    if proc_sexs[i]==0:
        root.set("gender","male")
    else:
        root.set("gender","female")

    # USER ID
    root.set("id", test_profsTOlikes[i])

    # EXTs
    root.set("extrovert","3.49")

    # NEUs
    root.set("neurotic","2.73")

    # AGRs
    root.set("agreeable","3.58")

    # CONs
    root.set("conscientious","3.45")

    # OPEs
    root.set("open", "3.91")

    # Create XML Tree
    tree = ET.ElementTree(root)

    # Write XML Tree to file
    tree.write(sys.argv[2]+"/"+str(user)+".xml")
    # tree.write("/Users/jamster/output/"+str(user)+".xml")




# def create_xmlA(i,user):
#     root = ET.Element("user")
#     if proc_ages[i]==1:
# 	    root.set("age_group","xx-24")
#     elif proc_ages[i]==2:
#         root.set("age_group","25-34")
#     elif proc_ages[i]==3:
#         root.set("age_group","35-49")
#     else:
#         root.set("age_group","50-xx")

#     if proc_sexs[i]==0:
#         root.set("gender","male")
#     else:
#         root.set("gender","female")

#     root.set("id", test_profsTOlikes[i])

#     root.set("extrovert","3.49")

#     root.set("neurotic", str(np.around(pred_neus[i], decimals=2)))

#     root.set("agreeable", str(np.around(pred_agrs[i], decimals=2)))

#     root.set("conscientious", str(np.around(pred_cons[i], decimals=2)))

#     root.set("open", str(np.around(pred_opes[i], decimals=2)))
#     tree = ET.ElementTree(root)
#     # tree.write(sys.argv[2]+"/"+str(user)+".xml")
#     tree.write("/Users/jamster/output/"+str(user)+".xml")

    


# Create XML files for users without LIKES data in the test dataset
###############
def create_xmlB(user):
    root = ET.Element("user")
    root.set("age_group","xx-24")
    root.set("gender","female")
    root.set("id", user)
    root.set("extrovert","3.49")
    root.set("neurotic","2.73")
    root.set("agreeable","3.58")
    root.set("conscientious","3.45")
    root.set("open","3.91")

    # Create XML Tree
    tree = ET.ElementTree(root)

    # Write XML Tree to file
    tree.write(sys.argv[2]+"/"+str(user)+".xml")
    # tree.write("/Users/jamster/output/"+str(user)+".xml")




# Generate the XML files from the predicted data
###############
for user in test_profilesLS:
    if user != "dummy" and user in test_profsTOlikes:
        ind = test_profsTOlikes.index(user)
        create_xmlA(ind, user)

        # if user in test_unqLikesUIDs:
        #     i2 = test_unqLikesUIDs.index(user)
        #     print(user, " ", tmp0[ind][0])
        #     print(user, " ", tmp0[i2][0])
    else:
        if user != "dummy":
            create_xmlB(user)



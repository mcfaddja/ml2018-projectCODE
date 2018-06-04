import readline
import scipy.sparse
import scipy
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
import xml.etree.ElementTree as ET
from sklearn.externals import joblib


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
for i in range(9500):
	profsTOlikes.append([])

for row in profilesLS:
	tmpIND = unqLikesUIDs.index(row[0])
	profsTOlikes[tmpIND]=row

profsTOlikes1=list(map(list, zip(*profsTOlikes)))
agesARR=np.array(profsTOlikes1[1])
sexsARR=np.array(profsTOlikes1[2])
opesARR=np.array(profsTOlikes1[3])
consARR=np.array(profsTOlikes1[4])
extsARR=np.array(profsTOlikes1[5])
agrsARR=np.array(profsTOlikes1[6])
neusARR=np.array(profsTOlikes1[7])

scores = {'randForr': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'adaBoost': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'bernNB': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
#'bagging': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'gradBoost': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'svm': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'linearSVM': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []}  }


attribs = [agesARR, sexsARR, opesARR, consARR, extsARR, agrsARR, neusARR]
labels = ['age', 'sex', 'ope', 'con', 'ext', 'agr', 'neu']

kf = KFold(n_splits=4)

cnt=0
for attrib in attribs:
    workARR = attrib
    label=labels[cnt]
    print(label)

    randForrC = RandomForestClassifier(n_jobs=7, n_estimators=15)
    randForrR = RandomForestRegressor(n_jobs=7, n_estimators=10)
    
    adaBoostC = AdaBoostClassifier(n_estimators=50)
    adaBoostR = AdaBoostRegressor(n_estimators=50)

    bernNB = BernoulliNB()
    gausRidge = linear_model.Ridge()
    
    #sdgC = linear_model.SGDClassifier()
    #sdgR = linear_model.SGDRegressor()
    
    #gradBoostC =  GradientBoostingClassifier(n_estimators=100, max_depth=1000)

    svmC = svm.SVC()
    svmR = svm.SVR()

    svmLc = svm.LinearSVC()
    svmLr = svm.LinearSVR()

    for train_index, test_index in kf.split(agesARR):
        trainX=likesMAT[train_index,:]
        yTrain=workARR[train_index]
        testX=likesMAT[test_index,:]
        yTest=workARR[test_index]

        print("start random forrest")
        if cnt < 2:
            randForrC.fit(trainX, yTrain)
            tmpSCR = randForrC.score(testX, yTest)
            scores['randForr'][label].append(tmpSCR)
        else: 
            randForrR.fit(trainX, yTrain)
            tmpSCR = randForrR.score(testX, yTest)
            scores['randForr'][label].append(tmpSCR)

        # print("start adaBoost")
        # adaBoostC.fit(trainX, yTrain)
        # tmpSCR = adaBoostC.score(testX, yTest)
        # scores['adaBoost'][label].append(tmpSCR)

        print("start bernoulli NB")
        if cnt < 2:
            bernNB.fit(trainX, yTrain)
            tmpSCR = bernNB.score(testX, yTest)
            scores['bernNB'][label].append(tmpSCR)
        else:
            gausRidge.fit(trainX, yTrain)
            tmpSCR = gausRidge.score(testX, yTest)
            scores['bernNB'][label].append(tmpSCR)

        # print("start gradient boost")
        # gradBoostC.fit(trainX, yTrain)
        # tmpSCR = gradBoostC.score(trainX, yTest)
        # scores['gradBoost'][label].append(tmpSCR)

        print("start SVM")
        if cnt < 2:
            svmC.fit(trainX, yTrain)
            tmpSCR = svmC.score(testX, yTest)
            scores['svm'][label].append(tmpSCR)
        else:
            svmR.fit(trainX, yTrain)
            tmpSCR = svmR.score(testX, yTest)
            scores['svm'][label].append(tmpSCR)

        print("start LINEAR svm")
        if cnt < 2:
            svmLc.fit(trainX, yTrain)
            tmpSCR = svmLc.score(testX, yTest)
            scores['linearSVM'][label].append(tmpSCR)
        else:
            svmLr.fit(trainX, yTrain)
            tmpSCR = svmLr.score(testX, yTest)
            scores['linearSVM'][label].append(tmpSCR)

    cnt+=1

print(scores)
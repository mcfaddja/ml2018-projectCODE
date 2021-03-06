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
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
import xml.etree.ElementTree as ET
from sklearn.externals import joblib


likes = pd.read_csv("/home/jamster/data/training/relation/relation.csv")

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


profilesDF=pd.read_csv("/home/jamster/data/training/profile/profile.csv")
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

scores = {'rand Forest': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'adaBoost': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'bern NB / gauss ridge': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'bagging (NO out of bag)': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'bagging (YES out of bag)': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'grad boost': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'stochastic grad descent': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'svm': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []},
'linear SVM': {'age': [], 'sex': [], 'ope': [], 'con': [], 'ext': [], 'agr': [], 'neu': []}  }


attribs = [agesARR, sexsARR, opesARR, consARR, extsARR, agrsARR, neusARR]
labels = ['age', 'sex', 'ope', 'con', 'ext', 'agr', 'neu']

kf = KFold(n_splits=10)

nJOBS = 7

cnt=0
for attrib in attribs:
    workARR = attrib
    label=labels[cnt]
    print(label)


    randForrC = RandomForestClassifier(n_jobs=nJOBS, n_estimators=500)
    randForrR = RandomForestRegressor(n_jobs=nJOBS, n_estimators=500)

    adaBoostC = AdaBoostClassifier(n_estimators=250)
    adaBoostR = AdaBoostRegressor(n_estimators=250)

    bagCoobN = BaggingClassifier(n_estimators=500, n_jobs=nJOBS)
    bagRoobN = BaggingRegressor(n_estimators=500, n_jobs=nJOBS)

    bagCoobY = BaggingClassifier(n_estimators=250, oob_score=True, n_jobs=nJOBS)
    bagRoobY = BaggingRegressor(n_estimators=250, oob_score=True, n_jobs=nJOBS)

    bernNB = BernoulliNB()
    gausRidge = linear_model.Ridge(max_iter=1e9, tol=1e-6)
    
    gradBoostC = GradientBoostingClassifier(n_estimators=500, max_depth=2500)
    gradBoostR = GradientBoostingRegressor(n_estimators=500, max_depth=2500)

    sdgC = linear_model.SGDClassifier(n_jobs=nJOBS)
    sdgR = linear_model.SGDRegressor()

    svmC = svm.SVC(tol=1e-6)
    svmR = svm.SVR(tol=1e-6)

    svmLc = svm.LinearSVC(tol=1e-6)
    svmLr = svm.LinearSVR(tol=1e-6)


    cntIN = 0
    for train_index, test_index in kf.split(agesARR):
        trainX=likesMAT[train_index,:]
        yTrain=workARR[train_index]
        testX=likesMAT[test_index,:]
        yTest=workARR[test_index]

        print("start random forrest")
        if cnt < 2:
            randForrC.fit(trainX, yTrain)
            tmpSCR = randForrC.score(testX, yTest)
            scores['rand Forest'][label].append(tmpSCR)
        else: 
            randForrR.fit(trainX, yTrain)
            tmpSCR = randForrR.score(testX, yTest)
            scores['rand Forest'][label].append(tmpSCR)

        print("start adaBoost")
        if cnt < 2:
            adaBoostC.fit(trainX, yTrain)
            tmpSCR = adaBoostC.score(testX, yTest)
            scores['adaBoost'][label].append(tmpSCR)
        else:
            adaBoostR.fit(trainX, yTrain)
            tmpSCR = adaBoostR.score(testX, yTest)
            scores['adaBoost'][label].append(tmpSCR)

        print("start bagging withOUT out-of-bag")
        if cnt < 2:
            bagCoobN.fit(trainX, yTrain)
            tmpSCR = bagCoobN.score(testX, yTest)
            scores['bagging (NO out of bag)'][label].append(tmpSCR)
        else:
            bagRoobN.fit(trainX, yTrain)
            tmpSCR = bagRoobN.score(testX, yTest)
            scores['bagging (NO out of bag)'][label].append(tmpSCR)

        print("start bagging WITH out-of-bag")
        if cnt < 2:
            bagCoobY.fit(trainX, yTrain)
            tmpSCR = bagCoobY.score(testX, yTest)
            scores['bagging (YES out of bag)'][label].append(tmpSCR)
        else:
            bagRoobY.fit(trainX, yTrain)
            tmpSCR = bagRoobY.score(testX, yTest)
            scores['bagging (YES out of bag)'][label].append(tmpSCR)

        print("start bernoulli NB / gauss ridge")
        if cnt < 2:
            bernNB.fit(trainX, yTrain)
            tmpSCR = bernNB.score(testX, yTest)
            scores['bern NB / gauss ridge'][label].append(tmpSCR)
        else:
            gausRidge.fit(trainX, yTrain)
            tmpSCR = gausRidge.score(testX, yTest)
            scores['bern NB / gauss ridge'][label].append(tmpSCR)

        print("start gradient boost")
        if cnt < 2:
            gradBoostC.fit(trainX, yTrain)
            tmpSCR = gradBoostC.score(testX, yTest)
            scores['grad boost'][label].append(tmpSCR)
        else:
            gradBoostR.fit(trainX, yTrain)
            tmpSCR = gradBoostR.score(testX, yTest)
            scores['grad boost'][label].append(tmpSCR)

        print("start stochastic gradient descent")
        if cnt < 2:
            sdgC.fit(trainX, yTrain)
            tmpSCR = sdgC.score(testX, yTest)
            scores['stochastic grad descent'][label].append(tmpSCR)
        else:
            sdgR.fit(trainX, yTrain)
            tmpSCR = sdgR.score(testX, yTest)
            scores['stochastic grad descent'][label].append(tmpSCR)

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
            scores['linear SVM'][label].append(tmpSCR)
        else:
            svmLr.fit(trainX, yTrain)
            tmpSCR = svmLr.score(testX, yTest)
            scores['linear SVM'][label].append(tmpSCR)

        cntIN+=1
        print(cntIN)
    cnt+=1

print(scores)

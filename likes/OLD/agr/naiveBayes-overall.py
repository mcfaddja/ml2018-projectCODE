import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import KFold



likes = pd.read_csv("~/data/training/relation/relation.csv")
profiles = pd.read_csv("~/data/training/profile/profile.csv")

likesDF = pd.DataFrame(likes)
profilesDF = pd.DataFrame(profiles)
likes = None
profiles = None

tempDF = pd.get_dummies(likesDF['userid'])
likesDF = None
print(tempDF)
tmpList = tempDF.columns.tolist()
print(tmpList[0])
#tempDF = tempDF.transpose()
#tempDF = pd.crosstab(index=likesDF["userid"], columns=likesDF["like_id"]).to_sparse()
likeDF = None
print(tempDF.shape)

fullDF = pd.merge(tempDF, profilesDF, on='userid')
print(fullDF)
#fullDF = pd.crosstab(index=fullDF["userid"], columns=[fullDF["age"], fullDF["gender"], fullDF["ope"], fullDF["con"], fullDF["ext"], fullDF["agr"], fullDF["neu"]], margins=True)
#fullDF = pd.crosstab(index=likesDF["userid"], columns=likesDF["like_id"])
#print(fullDF)
likesDF = None
tempDF = None
profilesDF = None


ageDFo = fullDF[['like_id', 'age']]
myMax = max(fullDF['age'])
ageDFo['age_grp'] = pd.cut(ageDFo['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])
#ageDFo['age_grp'] = pd.cut(ageDFo['age'], [0,25,35,50,myMax], right=False, labels=[1,2,3,4])
ageDF = ageDFo[['like_id', 'age_grp']]
ageDFo = None



genderDF = fullDF[['like_id', 'gender']]

opeDF = fullDF[['like_id', 'ope']]
conDF = fullDF[['like_id', 'con']]
extDF = fullDF[['like_id', 'ext']]
argDF = fullDF[['like_id', 'agr']]
neuDF = fullDF[['like_id', 'neu']]

fullDF = None

print(ageDF)




# likesTest = pd.read_csv("~/data/public-test-data/relation/relation.csv")
# profilesTest = pd.read_csv("~/data/public-test-data/profile/profile.csv")

# likesTestDF = pd.DataFrame(likesTest)
# profilesTestDF = pd.DataFrame(profilesTest)
# likesTest = None
# profilesTest = None

# fullTestDF = pd.merge(likesTestDF, profilesTestDF, on='userid')
# likesTestDF = None
# profilesTestDF = None

# print(fullTestDF)
# print(type(fullTestDF))

# ageTestDFo = fullTestDF[['like_id', 'age']]
# myTestMax = max(fullTestDF['age'])
# print(myTestMax)
# ageTestDFo['age_grp'] = pd.cut(ageTestDFo['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])
# ageTestDF = ageTestDFo[['like_id', 'age_grp']]
# ageTestDFo = None

# genderTestDF = fullTestDF[['like_id', 'gender']]

# opeTestDF = fullTestDF[['like_id', 'ope']]
# conTestDF = fullTestDF[['like_id', 'con']]
# extTestDF = fullTestDF[['like_id', 'ext']]
# argTestDF = fullTestDF[['like_id', 'agr']]
# neuTestDF = fullTestDF[['like_id', 'neu']]

# fullTestDF = None

# print(ageTestDF)



kf = KFold(n_splits=10)
print(kf.get_n_splits(ageDF))
for train_index, test_index in kf.split(ageDF):
    print("TRAIN: " + str(train_index) + "  TEST: " + str(test_index))


#gausNB = GaussianNB()
#like_id = ageDF['like_id']
#age_grp = ageDF['age_grp']
#gausNB.fit(like_id, age_grp)
#print(pd.isnull(ageDF))
#gausNB.fit(ageDF, ageDF['age_grp'])

bernNB = BernoulliNB()
bernNB.fit(ageDF["like_id"], ageDF["age_grp"])
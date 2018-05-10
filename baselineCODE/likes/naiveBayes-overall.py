import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB#, BernoulliNB, MultinomialNB



likes = pd.read_csv("~/data/training/relation/relation.csv")
profiles = pd.read_csv("~/data/training/profile/profile.csv")

likesDF = pd.DataFrame(likes)
profilesDF = pd.DataFrame(profiles)
likes = None
profiles = None

fullDF = pd.merge(likesDF, profilesDF, on='userid')
likesDF = None
profilesDF = None


ageDF = fullDF[['like_id', 'age']]
myMax = max(fullDF['age'])
ageDF['age_grp'] = pd.cut(ageDF['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])
ageDF = ageDF[['like_id', 'age_grp']]
#ageDFo = None

genderDF = fullDF[['like_id', 'gender']]

opeDF = fullDF[['like_id', 'ope']]
conDF = fullDF[['like_id', 'con']]
extDF = fullDF[['like_id', 'ext']]
argDF = fullDF[['like_id', 'agr']]
neuDF = fullDF[['like_id', 'neu']]

fullDF = None






likesTest = pd.read_csv("~/data/public-test-data/relation/relation.csv")
profilesTest = pd.read_csv("~/data/public-test-data/profile/profile.csv")

likesTestDF = pd.DataFrame(likesTest)
profilesTestDF = pd.DataFrame(profilesTest)
likesTest = None
profilesTest = None

fullTestDF = pd.merge(likesTestDF, profilesTestDF, on='userid')
likesTestDF = None
profilesTestDF = None


ageTestDFo = fullTestDF[['like_id', 'age']]
myTestMax = max(fullTestDF['age'])
print(myTestMax)
ageTestDFo['age_grp'] = pd.cut(ageTestDFo['age'], [0,25,35,50,myTestMax], right=False, labels=['xx-24','25-34','35-49','50+'])
ageTestDF = ageTestDFo[['like_id', 'age_grp']]
ageTestDFo = None

genderTestDF = fullTestDF[['like_id', 'gender']]

opeTestDF = fullTestDF[['like_id', 'ope']]
conTestDF = fullTestDF[['like_id', 'con']]
extTestDF = fullTestDF[['like_id', 'ext']]
argTestDF = fullTestDF[['like_id', 'agr']]
neuTestDF = fullTestDF[['like_id', 'neu']]

fullTestDF = None






gausNB = GaussianNB()
#like_id = ageDF['like_id']
#age_grp = ageDF['age_grp']
#gausNB.fit(like_id, age_grp)
gausNB.fit(ageDF['like_id'], ageDF['age_grp'])
print(gausNB.score(ageTestDF['like_id'], ageTestDF['age_grp']))
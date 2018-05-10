import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB



likes = pd.read_csv("~/data/training/relation/relation.csv")
profiles = pd.read_csv("~/data/training/profile/profile.csv")

likesDF = pd.DataFrame(likes)
profilesDF = pd.DataFrame(profiles)
likes = None
profiles = None

fullDF = pd.merge(likesDF, profilesDF, on='userid')
likesDF = None
profilesDF = None


ageDFo = fullDF[['like_id', 'age']]
myMax = max(fullDF['age'])
ageDFo['age_grp'] = pd.cut(ageDFo['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])
ageDF = ageDFo[['like_id', 'age_grp']]
ageDFo = None

genderDF = fullDF[['like_id', 'gender']]

opeDF = fullDF[['like_id', 'ope']]
conDF = fullDF[['like_id', 'con']]
extDF = fullDF[['like_id', 'ext']]
argDF = fullDF[['like_id', 'agr']]
neuDF = fullDF[['like_id', 'neu']]

fullDF = None

print(ageDF.shape)
print(neuDF.shape)

print(ageDF)
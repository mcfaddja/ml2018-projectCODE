import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# Gaussian Naive Bayes
#from sklearn import datasets
#from sklearn import metrics
#from sklearn.naive_bayes import GaussianNB

#dataset = datasets.load_iris()
#print(dataset.data)
#print(dataset.target)

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
print(myMax)
#ageDF = pd.DataFrame([fullDF['like_id']; pd.cut(fullDF['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])])
ageDFo['age_grp'] = pd.cut(ageDFo['age'], [0,25,35,50,myMax], right=False, labels=['xx-24','25-34','35-49','50+'])

ageDF = ageDFo[['like_id', 'age_grp']]
ageDFo = None

# ageDF = ageDFo.copy()
# ii = 0
# for row in ageDF.iterrows():
#     #tmpRow = row[1]
#     #age = tmpRow['age']
#     age = row[1]['age']
    
#     # print(age)
#     # print(row[0])
#     # print(type(age))
#     # print(row[1])

#     cat = 0
#     if age < 24.0:
#         cat = 0
#     elif age < 35.0:
#         cat = 1
#     elif age < 50:
#         cat = 2
#     else:
#         cat = 3

#     ageDF[ii,'age'] = cat
#     ii += 1

# ageDFo = None

print('age done')
genderDF = fullDF[['like_id', 'gender']]
print('gender done')
opeDF = fullDF[['like_id', 'ope']]
conDF = fullDF[['like_id', 'con']]
extDF = fullDF[['like_id', 'ext']]
argDF = fullDF[['like_id', 'agr']]
neuDF = fullDF[['like_id', 'neu']]

fullDF = None

print(ageDF.shape)
print(neuDF.shape)

print(ageDF)
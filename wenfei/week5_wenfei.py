import random
import numpy as np
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

path ="/data/training/text"
path_test= sys.argv[1]+"/text"
info = {}
info_test={}
files = os.listdir(path)
for file in files:
	index = file.rfind('.')
	name = file[:index]
	str=""
	with open(path+"/"+file, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			str = str+line
	info.update({name:str})

files = os.listdir(path_test)
for file in files:
	index = file.rfind('.')
	name = file[:index]
	str=""
	with open(path_test+"/"+file, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			str = str+line
	info_test.update({name:str})
#print (info_test)

# NB->age and gender.
df = pd.read_csv("/data/training/profile/profile.csv")
df['text']=''
for i in range(0,len(df)):
	#print(row['userid']+info.get(row['userid']))
	df.loc[i,'text'] = info.get(df.loc[i,'userid'])
	age = int(df.loc[i,'age'])
	if(age<=24):
		df.loc[i,'age']="xx-24"
	elif(age>=25 and age<=34):
		df.loc[i,'age']="25-34"
	elif(age>=35 and age<=49):
		df.loc[i,'age']="35-49"
	else:
		df.loc[i,'age']="50-xx"

data_profile = df.loc[:,['userid','age','gender','text']]
#print(data_profile)

df_test = pd.read_csv(sys.argv[1]+"/profile/profile.csv")
df_test['text']=''
for i in range(0,len(df_test)):
	df_test.loc[i,'text'] = info_test.get(df_test.loc[i,'userid'])

data_train = data_profile
data_test = df_test


count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'])
y_train = data_train['gender']
clf = MultinomialNB()
clf.fit(X_train, y_train)


#gender_NB
X_test = count_vect.transform(data_test['text'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)


for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'gender']=y_predicted

#gender_NB
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'])
y_train = data_train['age']
clf = MultinomialNB()
clf.fit(X_train, y_train)


#Age gender_NB
X_test = count_vect.transform(data_test['text'])
y_test = data_test['age']
y_predicted = clf.predict(X_test)

#print("age_Accuracy: %.2f" % accuracy_score(y_test,y_predicted))

for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'age']=y_predicted


user_list=df_test['userid']

def create_xml(i,user):
	root = ET.Element("user")
	root.set("age_group","xx-24")

	if(int(df_test['gender'][i])==0):
		root.set("gender","male")
	else:
		root.set("gender","female")
	root.set("id", df_test['userid'][i])
	root.set("extrovert","3.49")
	root.set("neurotic","2.73")
	root.set("agreeable","3.58")
	root.set("conscientious","3.45")
	root.set("open","3.91")
	tree = ET.ElementTree(root)
	tree.write(sys.argv[2]+"/"+user+".xml")

def get_index(user):
		for i in range(len(df_test.index)):
			if(df_test.loc[:,'userid'].values[i] == user):
				return i;

for user in user_list:
	create_xml(get_index(user),user)

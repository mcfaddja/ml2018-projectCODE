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
from sklearn.linear_model import LinearRegression
from sklearn import metrics

path ="/data/training/text"
path_test= sys.argv[1]+"/text"
info = {}
info_test={}
files = os.listdir(path)
for file in files:
	index = file.rfind('.')
	name = file[:index]
	st=""
	with open(path+"/"+file, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			st = st+line
	info.update({name:st})

files = os.listdir(path_test)
for file in files:
	index = file.rfind('.')
	name = file[:index]
	st=""
	with open(path_test+"/"+file, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			st = st+line
	info_test.update({name:st})
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

#five personality model
LIWC_data = pd.read_csv("/data/training/LIWC/LIWC.csv")
df_2 = pd.read_csv("/data/training/profile/profile.csv")
df_2.columns=['#','userId','age','gender','ope','con','ext','agr','neu']
all_data= pd.merge(df_2,LIWC_data)
big5=['ope','ext','con','agr','neu','#','userId','age','gender']#,'WPS','ppron','Dic','Numerals','funct','pronoun','article','verb']
feature_clos_ALL=[x for x in all_data.columns.tolist()[:] if not x in big5]
feature_clos=feature_clos_ALL

for i in range(0,len(df_2)):
	#print(row['userid']+info.get(row['userid']))
	age = int(all_data.loc[i,'age'])
	if(age<=24):
		all_data.loc[i,'age']= "xx-24"
	elif(age>=25 and age<=34):
		all_data.loc[i,'age']= "25-34"
	elif(age>=35 and age<=49):
		all_data.loc[i,'age']= "35-49"
	else:
		all_data.loc[i,'age']= "50-xx"


X_train = all_data[feature_clos]
y_train = all_data.neu
linreg = LinearRegression()
linreg.fit(X_train, y_train)

LIWC_data_test = pd.read_csv(sys.argv[1]+"/LIWC/LIWC.csv")
df_2_test = pd.read_csv(sys.argv[1]+"/profile/profile.csv")
#five personality prediction
df_2_test.columns=['#','userId','age','gender','ope','con','ext','agr','neu']
all_data_test= pd.merge(df_2_test,LIWC_data_test)
feature_clos_ALL_test=[x for x in all_data_test.columns.tolist()[:] if not x in big5]
feature_clos=feature_clos_ALL_test
X_test = all_data_test[feature_clos]
y_test = all_data_test.neu

y_predicted = linreg.predict(X_test)

for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'neu']=str(y_predicted)

y_train = all_data.ope
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_test = all_data_test.ope
y_predicted = linreg.predict(X_test)
for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'ope']=str(y_predicted)

y_train = all_data.ext
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_test = all_data_test.ext
y_predicted = linreg.predict(X_test)
for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'ext']=str(y_predicted)

y_train = all_data.con
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_test = all_data_test.con
y_predicted = linreg.predict(X_test)
for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'con']=str(y_predicted)


y_train = all_data.agr
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_test = all_data_test.agr
y_predicted = linreg.predict(X_test)
for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'agr']=str(y_predicted)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
y_train = all_data.age
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_test = all_data_test.age
y_predicted = logreg.predict(X_test)
for i,y_predicted in enumerate(y_predicted):
	df_test.loc[i,'age']=str(y_predicted)

user_list=df_test['userid']

def create_xml(i,user):
	root = ET.Element("user")
	root.set("age_group",df_test['age'][i])

	if(int(df_test['gender'][i])==0):
		root.set("gender","male")
	else:
		root.set("gender","female")
	root.set("id", df_test['userid'][i])
	root.set("extrovert",df_test['ext'][i])
	root.set("neurotic",df_test['neu'][i])
	root.set("agreeable",df_test['agr'][i])
	root.set("conscientious",df_test['con'][i])
	root.set("open",df_test['ope'][i])
	tree = ET.ElementTree(root)
	tree.write(sys.argv[2]+"/"+user+".xml")

def get_index(user):
		for i in range(len(df_test.index)):
			if(df_test.loc[:,'userid'].values[i] == user):
				return i;

for user in user_list:
	create_xml(get_index(user),user)

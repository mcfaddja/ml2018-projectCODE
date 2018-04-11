import pandas as pd
import numpy as np

df = pd.read_csv('/data/training/profile/profile.csv')
#print(df.shape)

personalities = [0.0, 0.0, 0.0, 0.0, 0.0]

for index, row in df.iterrows():
    personalities[0] += row['ope']
    personalities[1] += row['con']
    personalities[2] += row['ext']
    personalities[3] += row['agr']
    personalities[4] += row['neu']

personalitiesAVG = []
for row in personalities:
    tmp = row / 9500
    personalitiesAVG.append(tmp)

print(personalitiesAVG)
print("The average for OPE is:  " + str(personalitiesAVG[0]))
print("The average for CON is:  " + str(personalitiesAVG[1]))
print("The average for EXT is:  " + str(personalitiesAVG[2]))
print("The average for AGR is:  " + str(personalitiesAVG[3]))
print("The average for NEU is:  " + str(personalitiesAVG[4]))
import pandas as pd
import numpy as np

df = pd.read_csv('/data/training/profile/profile.csv')
print(df.shape)

#row = df.iloc[0]['gender']
#print(row)

maleCNT = 0
for index, row in df.iterrows():
    if row['gender'] == 0.0:
        maleCNT += 1

femaleCNT = 0
for index, row in df.iterrows():
    if row['gender'] == 1.0:
        femaleCNT += 1


print(maleCNT)
print(femaleCNT)

totalCNT = maleCNT + femaleCNT
print(totalCNT)
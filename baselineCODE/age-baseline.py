import pandas as pd
import numpy as np

df = pd.read_csv('/data/training/profile/profile.csv')
#print(df.shape)

minAge = [0.0, 25.0, 35.0, 50.0]
ageCNTs = {'0-24': 0, '25-34': 0, '35-49': 0, '50-inf': 0}
ageCNT = [0, 0, 0, 0]

for index, row in df.iterrows():
    if row['age'] < 25.0:
        ageCNT[0] += 1
        ageCNTs['0-24'] += 1
    elif row['age'] < 35:
        ageCNT[1] += 1
        ageCNTs['25-34'] += 1
    elif row['age'] < 50:
        ageCNT[2] += 1
        ageCNTs['35-49'] += 1
    else:
        ageCNT[3] += 1
        ageCNTs['50-inf'] += 1


#print(ageCNT)


if ageCNT.index(max(ageCNT)) == 0:
    print("The majority baseline age is:  0-24")
elif ageCNT.index(max(ageCNT)) == 1:
    print("The majority baseline age is:  25-34")
elif ageCNT.index(max(ageCNT)) == 2:
    print("The majority baseline age is:  34-49")
else:
    print("The majority baseline age is:  50-infinity")
    
    
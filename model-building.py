import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

url = 'car_evaluation.csv'
df = pd.read_csv(url)

categories = [df[i].unique() for i in df.columns]
categories[0] = categories[0][::-1]
categories[1] = categories[1][::-1]
categories[-1][2:] = categories[-1][2:][::-1]

oe = OrdinalEncoder(categories= categories)
df[df.columns] = oe.fit_transform(df[df.columns])

x = df.drop(['class'], axis = 1)
y = df['class']

rf = RandomForestClassifier(max_depth = 11,random_state = 48) 
rf.fit(x,y)

import pickle
pickle.dump(rf, open('car_evaluation_clf.pkl', 'wb'))

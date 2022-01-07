import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics

df=pd.read_csv("Covid Dataset.csv")
df=df.fillna(0)
df.shape
df=df.replace(["Yes","No"],[1,0])

x = df[['Breathing Problem','Fever','Dry Cough','Sore throat','Running Nose','Asthma','Headache','Abroad travel','Contact with COVID Patient','Attended Large Gathering','Visited Public Exposed Places','Family working in Public Exposed Places']]
y=df['COVID-19']
X= preprocessing.StandardScaler().fit(x).transform(x)

rfX_train, rfX_test, rfy_train, rfy_test = train_test_split(X, y, test_size=0.2, random_state=42)  # change in random state
print('Train set:', rfX_train.shape, rfy_train.shape)
print('Test set:', rfX_test.shape, rfy_test.shape)
from sklearn.ensemble import RandomForestClassifier

import math

clss = RandomForestClassifier(n_estimators=15, random_state=42)
clss.fit(rfX_train, rfy_train)
rftrain = clss.predict(rfX_train)
rfyhat = clss.predict(rfX_test)

import pickle
pickle.dump(clss, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
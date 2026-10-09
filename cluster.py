import pandas as pd
import numpy as np
import statistics
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("data.csv")
print(df)

x=df[['Year','Population_Growth','GDP_Growth','Sales_Growth','Technology_Growth']]

model=KMeans(n_clusters=7,random_state=0)
model.fit(x)

df['cluster']=model.labels_
print(df)

plt.scatter(df['Year'],df['Population_Growth'],c=df['cluster'])
plt.xlabel("year")
plt.ylabel("Population_growth")
plt.title("kmeans clusteing of growth dataset")
plt.show()



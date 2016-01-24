import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model




#Read in csv into dataframe
df = pd.read_csv(os.path.normpath("C:/Data/Stocks/aapl.csv"))

print(df.columns.values.tolist()) # put the each of the columns into an array of a multi-dimensional array
d = df["Close"] 
print(d[1])

closePrices = list(df.Close)
closePrices.reverse()

dayNum = [ np.reshape(x,-1) for x in np.arange(1,len(closePrices) + 1) ]

clf = linear_model.LinearRegression()
clf.fit (dayNum,closePrices)
print("\n\ny = " + str(clf.coef_[0]) + "x + " + str(clf.intercept_) + "\n\n")
print( clf.predict( [ [1] ] ) )

plt.plot(dayNum, closePrices)
plt.plot(dayNum,clf.predict(dayNum))
plt.show()


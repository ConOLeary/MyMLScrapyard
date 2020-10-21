# id:21-21--21
print("{############################################}\n")

####################################################################################################
#################### Data Processing ####################
import pandas as pd
df = pd.read_csv("week2.csv")
X1s = df['X1']
X2s = df['X2']
Ys = df['Y']
negX1s = []
negX2s = []
posX1s = []
posX2s = []
for i in range(len(df)):
    if(Ys[i] == 1):
        posX1s.append(X1s[i])
        posX2s.append(X2s[i])
    elif(Ys[i] == -1):
        negX1s.append(X1s[i])
        negX2s.append(X2s[i])

####################################################################################################
#################### Linear Regression ####################
from sklearn.linear_model import LinearRegression

indexes = range(0, len(df))
Xs = pd.DataFrame(X1s, indexes)
Ys = pd.DataFrame(X2s, indexes)
model = LinearRegression()
model.fit(Xs, Ys)
params = model.get_params(deep=True)
score = model.score(Xs, Ys, sample_weight=None)

####################################################################################################
#################### All Things Graphs ####################

import numpy as np
import matplotlib.pyplot as plt

#(iii)
predictions = model.predict(Xs)

print(predictions)


'''#(i)
np.random.seed(19680801) # Fixing random state for reproducibility
N = 50

x = negX1s
y = negX2s
colors = np.random.rand(N)
area = 100
plt.scatter(x, y, s=area, marker='o', c=colors, alpha=1)

x = posX1s
y = posX2s
colors = np.random.rand(N)
area = 100
plt.scatter(x, y, s=area, marker='+', c=colors, alpha=1)

plt.show()
'''

'''#(ii)
plt.scatter(Xs, Ys)
plt.plot(Xs, model.predict(Xs))
plt.show()
print("params: ",params)
print("score: ",score)
'''

print("\n{############################################}")
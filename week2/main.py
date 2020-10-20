# id:21-21--21

####################################################################################################
#################### Data Processing ####################
import pandas as pd
df = pd.read_csv("week2.csv")
print(df[['X1', 'X2', 'Y']])
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
        print(Ys[i])
    elif(Ys[i] == -1):
        negX1s.append(X1s[i])
        negX2s.append(X2s[i])
        print(Ys[i])



####################################################################################################
#################### All Things Graph ####################
import numpy as np
import matplotlib.pyplot as plt

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
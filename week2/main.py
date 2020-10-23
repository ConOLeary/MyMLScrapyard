# id:21-21--21
print("{############################################}\n")
'''
####################################################################################################
#################### Initial Data Processing ####################
import pandas as pd
import numpy as np
df = pd.read_csv("week2.csv")
X1s=df.iloc[:, 0] #or X1s = df['X1']
X2s=df.iloc[:, 1] #or X2s = df['X2']
Ys=df.iloc[:, 2] #or Ys = df['Y']
Xs = np.column_stack((X1s, X2s))
print(Xs)

####################################################################################################
#################### Linear Regression & Other Computation ####################
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Xs, Ys)

newYs = np.round(np.clip(model.predict(Xs), 0, 1))  # floats
model.fit(Xs, newYs)


negX1s = []
negX2s = []
posX1s = []
posX2s = []
newNegX1s = []
newNegX2s = []
newPosX1s = []
newPosX2s = []
for i in range(len(df)):
    if(Ys[i] == 1):
        posX1s.append(X1s[i])
        posX2s.append(X2s[i])
    elif(Ys[i] == -1):
        negX1s.append(X1s[i])
        negX2s.append(X2s[i])
    if(newYs[i] == 1):
        newPosX1s.append(X1s[i])
        newPosX2s.append(X2s[i])
    elif(newYs[i] == 0):
        newNegX1s.append(X1s[i])
        newNegX2s.append(X2s[i])

####################################################################################################
#################### All Things Graphs ####################
import matplotlib.pyplot as plt
light_g = (0.3, 0.7, 0.3)
hard_g = (0, 1, 0)
light_r = (0.7, 0.3, 0.3)
hard_r = (1, 0, 0)
np.random.seed(19680801) # Fixing random state for reproducibility
val_size = 200
prediction_size = 70

#(i + iii)
plt.figure()
plt.scatter(negX1s, negX2s, s=val_size, marker='o', c=light_g, alpha=0.9)
plt.scatter(posX1s, posX2s, s=val_size, marker='o', c=light_r, alpha=0.9)
plt.scatter(newNegX1s, newNegX2s, s=prediction_size, marker='*', c=hard_g, alpha=1)
plt.scatter(newPosX1s, newPosX2s, s=prediction_size, marker='*', c=hard_r, alpha=1)
plt.show()

#(ii)
#plt.scatter(Xs, Ys)
#plt.plot(Xs, model.predict(Xs))
#plt.show()
'''

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd

iris = datasets.load_iris()
# Select 2 features / variable for the 2D plot that we are going to create.
df = pd.read_csv("week2.csv")
X1s=df.iloc[:, 0] #or X1s = df['X1']
X2s=df.iloc[:, 1] #or X2s = df['X2']
Ys=df.iloc[:, 2] #or Ys = df['Y']
Xs = np.column_stack((X1s, X2s))
X = Xs
y = Ys
#X = iris.data[:, :2]  # we only take the first two features.
#y = iris.target

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

print("\n{############################################}")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import eigh

from google.colab import drive
drive.mount('/content/drive')

"""# Read and Load Data"""

from scipy.io import loadmat
webcam = loadmat('webcam.mat')
dslr = loadmat('dslr.mat')
amazon = loadmat('amazon.mat')
caltech = loadmat('caltech10.mat')



def loadX(data):
  X = data['fts']

  return X

def loady(data):
  y = data['labels']

  return y

"""# Scaling"""

from sklearn.preprocessing import StandardScaler

def standarized(X):
  """
  Return normalized features.
 
  """
  X = StandardScaler().fit_transform(X)

  return X

"""# PC"""

def PC(X,d):
  """
  Return d principle components with highest variance

  """
  cov_mat = np.cov(X.T)
  eig_vals, eig_vecs = eigh(cov_mat)
  components = np.column_stack((eig_vecs[:,-i] for i in range(1,d+1)))

  #n_components with highest variance
  # var_exp = [(i / sum(eig_vals))*100 for i in sorted(eig_vals, reverse=True)]
  # var_exp = np.cumsum(var_exp)

  # import matplotlib.pyplot as plt
  # plt.plot( var_exp)
  # plt.xlabel('Number of components')
  # plt.ylabel('Variance') 
  # plt.show()

  return components

"""# Task 1.1

Subspace Alignment
"""

def subal(S,T,d):
  Xs = PC(S, d)
  Xt = PC(T, d)

  # Defining the alignment matrix
  M = np.dot(Xs.T, Xt)

  # Computing Xa
  Xa = np.dot(Xs, M)

  # Computing source and target projected data 
  Sa = np.dot(S, Xa)
  Ta = np.dot(T, Xt)

  # Fitting a 1-NN classifier 
  KNN = KNeighborsClassifier(n_neighbors=1)
  KNN.fit(Sa,yS)
  pred= KNN.predict(Ta)

  # Accuracy
  print(accuracy_score(yT, pred))

"""# Task 1.2

Webcam being the source and dslr being target
"""

# Load source and target features
S = loadX(webcam)
T = loadX(dslr)

# Load Source and target labels
yS = loady(webcam)
yT = loady(dslr)

# Scaling source and target features
S = standarized(S)
T = standarized(T)

subal(S,T,96)

"""Dslr being the source and webcam being target"""

# Load source and target features
S = loadX(dslr)
T = loadX(webcam)

# Load Source and target labels
yS = loady(dslr)
yT = loady(webcam)

subal(S,T,96)

"""# Task 2.1

Sinkhorn- knopp
"""

pip install POT

import ot
import scipy
from scipy.spatial import distance

def sinkhorn(S,T,reg_e):
  a = np.ones(S.shape[0])
  b = np.ones(T.shape[0])

  # Loss matrix
  M = scipy.spatial.distance.cdist(S,T)


  #Normalize M
  from sklearn import preprocessing
  M_norm = preprocessing.normalize(M,"max")

  #Coupling matrix
  G = ot.sinkhorn(a,b,M_norm, reg_e)
  Sa = np.dot(G,T)
  Sa = Sa.astype(np.float64)

  # Fitting a 1-NN classifier
  from sklearn.neighbors import KNeighborsClassifier
  KNN = KNeighborsClassifier(n_neighbors=1)
  KNN.fit(Sa, yS)
  pred = KNN.predict(T)
  print(accuracy_score(yT, pred))

"""# Task 2.2

Webcam being the source and dslr being the target
"""

# Load source and target features
S = loadX(webcam)
T = loadX(dslr)

# Load Source and target labels
yS = loady(webcam)
yT = loady(dslr)

# Scaling source and target features
S = standarized(S)
T = standarized(T)

sinkhorn(S,T,0.01)

"""Dslr being the source and webcam being the target"""

# Load source and target features
S = loadX(dslr)
T = loadX(webcam)

# Load Source and target labels
yS = loady(dslr)
yT = loady(webcam)

# Scaling source and target features
S = standarized(S)
T = standarized(T)

sinkhorn(S,T,0.01)


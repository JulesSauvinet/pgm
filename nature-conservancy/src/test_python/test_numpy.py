# Any results you write to the current directory are saved as output.

import numpy as np


d = np.random.random_sample((1000, 10))
print("np.shape(d) ", np.shape(d))


others = d[:,5:8].sum(axis=1)
others = np.reshape(others,(others.shape[0],1))
b = np.concatenate((d[:,0:5], others, d[:,8:10]), axis=1)

#d = np.concatenate((d[0:1],[d[1:3].sum(axis=0)],d[3:4]),axis=0)

print("np.shape(b) ", np.shape(b))

#print(d.tolist())




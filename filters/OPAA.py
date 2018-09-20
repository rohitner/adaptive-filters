import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def OPAA(x,d,y,n,s,xi):
  a = np.zeros([n,d])
  a[0] = np.dot(np.linalg.inv(np.dot(x[0:s].T,x[0:s])),np.dot(x[0:s].T,y[0:s]))
  for t in range(1,n):
    if abs(y[t] - np.dot(a[t],x[t])) < xi:
      loss = 0
    else:
      loss = abs(y[t] - np.dot(a[t],x[t])) - xi
    tau = loss/(np.linalg.norm(x[t])**2)
    a[t] = a[t-1] + np.dot(np.dot(np.sign(y[t] - np.dot(a[t-1],x[t])),x[t]),tau)
  return a

x, a, yc, y = data.generate_train_set()
b = OPAA(x,4,y,600,20,0.5)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='OPAA' ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Online Passive Agressive Algorithm')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def MWLS(x,y,n,w):
  a = np.zeros([n,4])
  for i in range(w):
    a[i] = np.dot(np.linalg.inv(np.dot(x[0:w-1].T,x[0:w-1])),np.dot(x[0:w-1].T,y[0:w-1]))
  for t in range(w,n):
    a[t] = np.dot(np.linalg.inv(np.dot(x[t-w:t].T,x[t-w:t])),np.dot(x[t-w:t].T,y[t-w:t]))
  return a

x, a, yc, y = data.generate_train_set()
b = MWLS(x,y,600,50)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='MWLS'  ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Moving Window Least Squares')
plt.legend()
# plt.show()

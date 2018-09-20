import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def KF(X,d,Y,n):
  var = 10
  d = 4
  b = np.zeros((600,4))
  bk = np.ones([d,1])
  P = np.identity(d)
  Q = np.identity(d)*0.1 #process noise

  for i in range(1, n):
    x = np.reshape(X[i],(1,4))
    em = np.random.normal(0, var) # measurement noise
    y = Y[i]
    P = P + Q
    R = em**2
    S = np.dot(x,np.dot(P,x.T)) + R
    K = np.dot(np.dot(P,x.T),np.linalg.inv(np.reshape(S,(1,1))))
    bk = bk + np.dot(K,y - np.dot(x,bk))
    b[i] = np.reshape(bk,4)
    P = np.dot(np.identity(d)-np.dot(K,np.reshape(x,(1,4))),P)
  return b

x, a, yc, y = data.generate_train_set()
b = KF(x,4,y,600)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='KF'  ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Kalman Filter')
plt.legend()
# plt.show()

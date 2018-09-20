import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def PLS(x,d,y,n,l=3):
  a = np.zeros([600,4])
  TAU = []
  p = []
  C = []
  Yhat = []

  for t in range(1,n):
    X = x[0:t]
    Y = y[0:t]
    XC = X
    YC = Y
    for i in range(0,l):
      W = np.dot(X.T,Y)/np.linalg.norm(np.dot(X.T,Y))
      Tau = np.reshape(np.dot(X,W),[t,1])
      P = np.reshape(np.dot(X.T,Tau)/np.dot(Tau.T,Tau),[d,1])
      X = X - np.dot(Tau,P.T)
      c = np.dot(Tau.T,Y)/np.dot(Tau.T,Tau)
      Y = np.reshape(Y,[t,1]) - np.dot(Tau,c)
      if i == 0:
        TAU = Tau
        p = P
        C = np.append(C,c)
      else:
        TAU = np.append(TAU,Tau,axis=1)
        p = np.append(p,P,axis=1)
        C = np.append(C,c)
      a[t] = a[t] + np.reshape(W*c,4)
  return a

x, a, yc, y = data.generate_train_set()
b = PLS(x,4,y,600)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='PLS'  ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Partial Least Squares')
plt.legend()
# plt.show()

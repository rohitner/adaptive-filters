import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def RLS(x,d,y,n,mu,s):
  a = np.zeros([n,d])
  a[0] = np.dot(np.linalg.inv(np.dot(x[0:s].T,x[0:s])),np.dot(x[0:s].T,y[0:s]))
  P = np.zeros([n,d,d])
  P[0] = np.linalg.inv(np.dot(x[0:s].T,x[0:s]))
  for t in range(1,n):
    xt = np.reshape(x[t],[1,d])
    e = y[t]-np.dot(xt,np.reshape(a[t-1],[d,1]))
    k = np.dot(P[t-1],xt.T)/(mu + np.linalg.multi_dot([xt,P[t-1],xt.T]))
    a[t] = a[t-1] + np.dot(k,e).T
    P[t] = (1/mu)*(P[t-1]-np.linalg.multi_dot([k,xt,P[t-1]]))
  return a

x, a, yc, y = data.generate_train_set()
b = RLS(x,4,y,600,0.94,20)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='RLS'  ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Recursive Least Squares')
plt.legend()
# plt.show()

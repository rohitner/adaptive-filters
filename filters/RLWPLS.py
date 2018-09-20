import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(5)

def spcwt(X,xq):
  d = [np.linalg.norm(xi - xq) for xi in X]
  w = np.zeros(len(X))
  for i in range(len(d)):
    w[i] = (d[i]-np.amin(d))/(np.amax(d)-np.amin(d))
  return w

def RLWPLS(x,y,mod=0):
  M = 3       # tap size
  w = 40      # initial window size
  wmin = 41   # max window size
  wmax = 70   # min window size
  s = 30      # space window size (<w)
  muc = 0.8   # FF for space window
  mu = 0.95   # FF for time window
  l = 3       # number of latent variables for PLS
  rho = 0.8   # space weight for time window
  a = np.zeros([600,4])
  YQ = np.dot(x,np.linalg.multi_dot([np.linalg.inv(np.dot(x.T,x)),x.T,y]))
  for t in range(w,600):
    X = x[t-w:t]
    Y = y[t-w:t]
    xq = x[t]
    for z in range(0,l):
      WT = np.diag(np.append(np.array([muc for i in range(0,s)]),np.array([mu**(w-i-1) for i in range(s,w)])))
      WS = np.diag(np.append(np.array(spcwt(X,xq)[0:s]),np.array([rho for i in range(s,w)])))
      W  = np.dot(WS,WT)
      yb = np.dot(np.diag(W),Y) / np.sum(np.diag(W))
      yq = yb
      uk = np.linalg.multi_dot([X.T,W,Y])
      uk = uk / np.linalg.norm(uk)
      tk = np.linalg.multi_dot([X,uk])
      pk = np.linalg.multi_dot([tk.T,W,X]) / np.linalg.multi_dot([tk.T,W,tk])
      qk = np.linalg.multi_dot([tk.T,W,Y]) / np.linalg.multi_dot([tk.T,W,tk])
      Tk = np.linalg.multi_dot([xq,uk])
      X  = X - np.dot(np.reshape(tk,[w,1]),np.reshape(pk,[1,4]))
      Y  = Y - np.dot(tk,qk)
      xq = xq - np.dot(Tk,pk)
      yq = yq + np.dot(Tk,qk)
      YQ[t] = yq
      a[t] = a[t] + np.dot(uk,qk)
  return a

x, a, yc, y = data.generate_train_set()
b = RLWPLS(x,y)

for i in range(4):
  plt.subplot(int('22'+str(i+1)))
  plt.plot(a[:,i],label='truth',color='black')
  plt.plot(b[:,i],label='RLWPLS'  ,color='red')
  plt.title("Dimension "+str(i))

plt.suptitle('Recursive Locally Weighted Partial Least Squares')
plt.legend()
# plt.show()

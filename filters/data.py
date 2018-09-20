import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

def generate_train_set():
  a = np.zeros([600,4])
  for i in range(0,600):
    if i < 200:
      a[i,0] = -1.0 + i/200.0
      a[i,1] = +0.5
      a[i,2] = +1.0 - i*0.75/200
      a[i,3] = +0.25
    elif 200 <= i < 400:
      a[i,0] = 0.0
      a[i,1] = -0.5
      a[i,2] = +0.25
      a[i,3] = +1.0
    elif i >= 400:
      a[i,0] = a[i-1,0] + 1/200.0
      a[i,1] = +0.75
      a[i,2] = a[i-1,2] - 1/200.0
      a[i,3] = -0.5

  x = np.zeros([600,4])
  yc = np.zeros(600)
  y = np.zeros(600)
  u = np.random.uniform(-2,2,600)

  for t in range(1,600):
    x[t] = [1, yc[t-1], u[t], u[t-1]]
    yc[t] = np.dot(a[t],x[t])
    noise = np.random.normal(0,0.4)
    y[t] = yc[t] + noise

  return x, a, yc, y
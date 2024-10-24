import numpy as np


num = 100
residule = 0
A = np.random.rand(num,num)
A = np.dot(A,A.T)
b = np.random.rand(num,1)
x = b
# x = b/np.linalg.norm(b)
xk = x
rk = b - np.dot(A,xk)
pk = rk
step = 0
while(1):
    low = np.dot(rk.T,rk)
    Ap = np.dot(A,pk)
    ak = low/np.dot(pk.T,Ap)
    xk = xk + ak*pk
    rk = rk - ak*Ap
    up = np.dot(rk.T,rk)
    if(np.sqrt(up) < 1e-6):
        break
    bk = up / low
    pk = rk + bk*pk
    # pk = pk1
    # rk = rk1
    # xk = xk1
    step+=1
print("Residule is ", np.linalg.norm(b-np.dot(A,xk)), step)



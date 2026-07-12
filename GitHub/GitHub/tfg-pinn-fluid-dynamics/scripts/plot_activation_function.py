import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-0.9,0.9,100000)

def f(x):
    return np.where(x>=0,1,0)
    #return 1/(1+np.exp(-x))
    #return np.where(x>=0,x,0)
    #return np.tanh(x)


y=f(x)
plt.figure(figsize=(8, 6))
plt.plot(x,y)
plt.xlim(-1,1)
#plt.ylim(-1.1,1.1)
plt.tick_params(labelsize=16)
plt.xlabel('x',fontsize=17)
plt.ylabel('f(x)',labelpad=15,fontsize=17,rotation=0)
plt.title('Escalón',fontsize=17)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('matrices_t.txt',delimiter =',')
# Renaming columns for better readability
t = data[:,0]
x = data[:,1]
u = data[:,2]

plt.plot(x,u)

def solucion(x,t):
  parte0 = np.where(x<=-0.15,0.0,0.0)
  parte1 = np.where(np.logical_and(-0.15<=x, x<=t-0.15),(x+0.15)/t,0.0)
  parte2 = np.where(np.logical_and(t-0.15<= x, x<= t/2+0.15),1.0,0.0)
  parte3 = np.where(x>=t/2+0.15,0.0,0.0)
  return parte0+parte1+parte2+parte3



puntos=500
xmin=-0.5
xmax=0.5
x = np.linspace(xmin,xmax,puntos)

t=0.2
# Evalúa la función real y la salida de la red neuronal en la malla de puntos
t_array=np.zeros(puntos)
Z_real=np.zeros(puntos)
for i in range (0,puntos):
    Z_real[i] = solucion(x[i],t)
    t_array[i]=t
plt.plot(x, Z_real, label='Solución Analítica',color='r',linewidth=2.5)

import numpy as np
import matplotlib.pyplot as plt

# Definimos el rango de tiempo y espacio
x2 = np.linspace(-0.15, 0.15, 400)
t = np.linspace(0.01, 0.6, 400)  # Evitamos t=0 para no tener divisiones por cero
X, T = np.meshgrid(x2, t)

y=x2/t



# Configuración de la gráfica
plt.figure(figsize=(8, 6))


# Líneas características
for x in np.arange(-0.55, -0.1, 0.05):
    plt.axvline(x=x, color='b', linestyle='-', lw=1)

for x in np.arange(0, 0.45, 0.05):
    plt.plot(x2 + x, t, color='b', linestyle='-', lw=1)

for x in np.arange(0.15, 0.65, 0.05):
    plt.axvline(x=x, color='b', linestyle='-', lw=1)

# Crear las pendientes para las rectas de rarefacción (rojas)
slopes_red = np.linspace(0.05, 0.45, 6)

# Dibujar las rectas de rarefacción (rojas)
for s in slopes_red:
    plt.plot(s * t-0.15, t, 'r')

# Etiquetas y límites

plt.xticks(np.arange(-0.6, 0.8, 0.15))
plt.xlim(-0.5, 0.5)
plt.ylim(0, 0.6)
plt.xlabel('x', fontsize=17)
plt.ylabel('t',labelpad=15,fontsize=17,rotation=0)
plt.title('Características y ondas de rarefacción', fontsize=17)

plt.tick_params(labelsize=16)



plt.legend(fontsize=13)
plt.show()

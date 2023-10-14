import numpy as np
import matplotlib.pyplot as plt

# Gerar pontos no plano XY
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Equações das retas
Z1 = 1 - 2*X - Y
Z2 = 3*X + 4*Y - 1

# Plotar as retas
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z1, levels=[0], colors='r', label='2x + y + z = 1')
plt.contour(X, Y, Z2, levels=[0], colors='b', label='3x + 4y - z = 1')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Gráfico das Equações Lineares')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

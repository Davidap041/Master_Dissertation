import control
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Sistema de Integrador Unico 2 agentes
'''
Momento 1 - Sistema andando com velocidade contante em d1 e d6 para situação de não ultrapassagem

Momento 2 - Caso de ultrapassagem 
Sistema recebe ponto inicial em d6 = 1 e d3 = 2, e a referência 0 é o ponto em d2
'''
# Momento 1 - 
X0_initial_m2 = np.array([2, 2, 0])
X0_initial_m4 = np.array([-2, -2, 0])

a31 = 0.5
a12 = 0.5
a21 = 0.5
a32 = 0.5

# Matriz de Adjacência
Adj = np.array([[0, a12,0], [a21, 0,0],[a31,a32,0]])
print("Matriz de Adjacencia")
print(Adj)
AdjT = Adj.T

# Matriz Diagonal
D = np.array([
    [np.sum(AdjT[0]),0,0],
    [0, np.sum(AdjT[1]),0],
    [0, 0,np.sum(AdjT[2])],
    ])
print("Matriz Diagonal")
print(D)

# Matriz Laplaciana
L = D - Adj
print("Matriz Laplaciana")
print(L)

# Eigenstructure of Graph Laplacian Matrix
L = np.matrix(L)
autovalores_Laplaciano, autovetores_Laplaciano = np.linalg.eig(L)
print("Autovetores da Matriz Laplaciana")
print(autovetores_Laplaciano)
print("AutoValores da Matriz Laplaciana")
print(autovalores_Laplaciano)
# Como calular a Forma de Jordan em Python ??

# "Sistema" Laplaciano resultante
A_Laplace = -L
B_Laplace = np.array([[0.8], [0.2],[0]])
C_Laplace = np.eye(3)
D_Laplace = 0
sys_Laplace = control.ss(A_Laplace, B_Laplace, C_Laplace, D_Laplace)
print("Sistema Laplaciano")
print(sys_Laplace)
# Convert velocity-based system to position-based system
A_pos = np.eye(3)  # Identity matrix for positions
B_pos = np.array([[0.8], [0.2], [0]])
C_pos = np.eye(3)
D_pos = 0
sys_pos = control.ss(A_pos, B_pos, C_pos, D_pos)

# Display the Laplacian matrix for the position-based system
print("Laplacian Matrix for Position-based System")
print(A_pos - B_pos @ np.linalg.pinv(C_pos) @ B_pos.T)

# Simuation Time
t = np.arange(0, 20, 0.1)
# Simulationa Initial
t1_lap_initial, y1_lap_initial, x1_lap_initial = control.initial_response(
    sys_Laplace, T=t, X0=X0_initial_m2, return_x=True)
# Simulation Impulse
t2_lap_initial, y2_lap_initial, x2_lap_initial = control.initial_response(
    sys_Laplace, T=t, X0=X0_initial_m4, return_x=True)

# Figures to Simulation Initial
plt.figure()
plt.title("Resposta a condicao Inicial Momento 2 para o 3")
plt.plot(t1_lap_initial, x1_lap_initial[0, :])
plt.plot(t1_lap_initial, x1_lap_initial[1, :])
# plt.plot(t1_lap_initial, x1_lap_initial[2, :])
plt.xlabel("tempo (s)")
plt.ylabel("Posição")
plt.legend(["Agente A", "Agente B"])
plt.grid()

# # Figures to Simulation Impulse
plt.figure()
plt.title("Resposta a condicao Inicial Momento 4 para o 5")
plt.plot(t2_lap_initial, x2_lap_initial[0, :])
plt.plot(t2_lap_initial, x2_lap_initial[1, :])
# plt.plot(t1_lap_initial, x1_lap_initial[2, :])
plt.xlabel("tempo (s)")
plt.ylabel("Posição")
plt.legend(["Agente A", "Agente B"])
plt.grid()

# show figure
plt.show()

# Calculo da constante de tempo (Frank)
tau_consensus = 1/autovalores_Laplaciano[1]
print("Constante de tempo do Integrador Unico")
print(tau_consensus)

# Calculo do Valor final de consenso (Frank)
# right eigenvector | # left eigenvector
# A*x     = lx      | # x*A  =  x*l
# A*x -lx = 0       | # 0    =  xl -xA
# (A -l)x = 0       | # 0    =  x(l-A)

# w1 = [p1 ... pn].T is the normalized left eigenvector of the Laplacian L for lambda1 = 0
# c = sum(pi*xi(0)); xi:= Initial condition
w = scipy.linalg.eig(L, left = True, right = False)[1]

# não consegui fazer a multiplicação de matrizes!!!
# w0 = np.transpose(w[0,:])
w0 = w[1,:].T
print("w0(transposta):\n ", w0)
#c = np.multiply(w0,X0_initial)
c = np.multiply(w0.T,X0_initial_m2)
print("Matriz C:\n", c)
print("Valor de consenso:\n" , c.sum())

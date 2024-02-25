import control
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Create a "fig" directory if it doesn't exist
import os
if not os.path.exists('fig'):
    os.makedirs('fig')

# Sistema de Integrador Unico 2 agentes e o lider
'''
Momento 1 - Não ocorre ultrapassagem
a velocidade entre os agentes é mantida e a referência também

'''

X0_initial_m2 = np.array([0, 0, 0])

a31 = 0.5
a12 = 0.5
a21 = 0.5
a32 = 0.5

# Matriz de Adjacência
Adj = np.array([[0, a12, 0], [a21, 0, 0], [a31, a32, 0]])
print("Matriz de Adjacencia")
print(Adj)
AdjT = Adj.T

# Matriz Diagonal
D = np.array(
    [
        [np.sum(AdjT[0]), 0, 0],
        [0, np.sum(AdjT[1]), 0],
        [0, 0, np.sum(AdjT[2])],
    ]
)
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
B_Laplace = np.array([[1], [0], [0]])
C_Laplace = np.eye(3)
D_Laplace = 0
sys_Laplace = control.ss(A_Laplace, B_Laplace, C_Laplace, D_Laplace)
print("Sistema Laplaciano")
print(sys_Laplace)


# Calculo da constante de tempo (Frank)
tau_consensus = 1 / autovalores_Laplaciano[1]
print("Constante de tempo do Integrador Unico")
print(tau_consensus)

# Calculo do Valor final de consenso (Frank)
# right eigenvector | # left eigenvector
# A*x     = lx      | # x*A  =  x*l
# A*x -lx = 0       | # 0    =  xl -xA
# (A -l)x = 0       | # 0    =  x(l-A)

# w1 = [p1 ... pn].T is the normalized left eigenvector of the Laplacian L for lambda1 = 0
# c = sum(pi*xi(0)); xi:= Initial condition
w = scipy.linalg.eig(L, left=True, right=False)[1]

# não consegui fazer a multiplicação de matrizes!!!
# w0 = np.transpose(w[0,:])
w0 = w[1, :].T
print("w0(transposta):\n ", w0)
# c = np.multiply(w0,X0_initial)
c = np.multiply(w0.T, X0_initial_m2)
print("Matriz C:\n", c)
print("Valor de consenso:\n", c.sum())

'''
Gráficos
'''

# Simuation Time
t = np.arange(0, 20, 0.1)
# Simulationa Initial
t1_lap_initial, y1_lap_initial, x1_lap_initial = control.initial_response(
    sys_Laplace, T=t, X0=X0_initial_m2, return_x=True
)

# Figures to Simulation Initial
plt.figure()
plt.title("Resposta a condicao Inicial Momento 1")
plt.plot(t1_lap_initial, x1_lap_initial[0, :])
plt.plot(t1_lap_initial, x1_lap_initial[1, :])
# plt.plot(t1_lap_initial, x1_lap_initial[2, :])
plt.xlabel("tempo (s)")
plt.ylabel("Posição")
plt.legend(["Agente A", "Agente B"])
plt.grid()
plt.savefig('fig/case6_fig3.png', bbox_inches='tight')

# Mark the last values on the plot
last_value_index = -1
last_value_A = x1_lap_initial[0, last_value_index]
last_value_B = x1_lap_initial[1, last_value_index]

plt.scatter(t1_lap_initial[last_value_index], last_value_A, color='red', marker='o')
plt.scatter(t1_lap_initial[last_value_index], last_value_B, color='blue', marker='o')

# Display the last values
plt.text(t1_lap_initial[last_value_index], last_value_A, f'A: {last_value_A:.2f}', ha='right', va='bottom')
plt.text(t1_lap_initial[last_value_index], last_value_B, f'B: {last_value_B:.2f}', ha='right', va='top')

# Define the time array
t = np.arange(0, 120, 0.1)

# Define a function for force F that varies over time
def force_function(t):
    F = np.ones_like(t)  # Initialize with ones
    # Change force values at different time intervals
    F[t < 30] = 0.2  # Force is 1.5 for t < 5
    F[(t >= 30) & (t < 60)] = 0.5  # Force is 0.8 for 5 <= t < 10
    F[(t >= 60) & (t < 90)] = 0.2  # Force is 1.2 for 10 <= t < 15
    F[t >= 90] = 0  # Force is 1.0 for t >= 15
    return F

# Evaluate the force function for the given time array
F = force_function(t)
# F = np.ones_like(t) * 1 # Step input with value 2

# Simulate the system response to the forced input
t_forced, y_forced, x_forced = control.forced_response(sys_Laplace, t, F, X0_initial_m2,return_x=True)

# Extract individual states
x1 = x_forced[0, :]
x2 = x_forced[1, :]
x3 = x_forced[2, :]

# Plot the forced response
plt.figure(figsize=(12, 6))

# Plot x1, x2, and y on the same graph
plt.plot(t_forced, x1, label='Agent A')
plt.plot(t_forced, x2, label='Agent B')
#plt.plot(t_forced, x3, label='Leader')

# Add labels and legend
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Position')
plt.legend()
plt.savefig('fig/case7_fig1.png', bbox_inches='tight')

# Simulate the system response to the forced input
t_forced, y_forced, x_forced = control.forced_response(sys_Laplace, t, F, X0_initial_m2, return_x=True)

# Extract individual states
x1 = x_forced[0, :]
x2 = x_forced[1, :]
x3 = x_forced[2, :]

# Calculate velocity
dt = t_forced[1] - t_forced[0]
v1 = np.gradient(x1, dt)
v2 = np.gradient(x2, dt)
v3 = np.gradient(x3, dt)

# Plot the forced response with velocity
plt.figure(figsize=(10, 6))

# Plot position and velocity on the same graph
#plt.plot(t_forced, x1, label='Agent A (Position)')
plt.plot(t_forced, v1, label='Agent A (Velocity)', color='blue')

# plt.plot(t_forced, x2, label='Agent B (Position)')
plt.plot(t_forced, v2, label='Agent B (Velocity)', color='magenta')

# plt.plot(t_forced, x3, label='Leader (Position)')
#plt.plot(t_forced, v3, label='Leader (Reference)', color='red')

# Mark the last values on the plot
# last_value_A = v1[0, last_value_index]
# last_value_B = v2[1, last_value_index]


#plt.scatter(t_forced[last_value_index], last_value_A, color='red', marker='o')
#plt.scatter(t_forced[last_value_index], last_value_B, color='blue', marker='o')
#plt.scatter(t_forced[last_value_index], last_value_C, color='green', marker='o')

# Display the last values
#plt.text(t_forced[last_value_index], last_value_A, f'A: {last_value_A:.2f}', ha='right', va='bottom')
#plt.text(t_forced[last_value_index], last_value_B, f'B: {last_value_B:.2f}', ha='right', va='top')
#plt.text(t_forced[last_value_index], last_value_C, f'C: {last_value_C:.2f}', ha='right', va='top')


# Add labels and legend
plt.xlabel('Time (min)')
plt.ylabel('Position / Velocity')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

# Add grid

# Save the plot as an image in the "fig" directory
plt.grid()
plt.savefig('fig/case7_fg2.png', bbox_inches='tight')
#plt.show()

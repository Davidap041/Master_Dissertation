import control
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Sistema de Integrador Unico 2 agentes e o lider
# Momento 1 
# Momento 2
X0_initial_m2 = np.array([2, -1, 0])
X0_initial_m4 = np.array([-2, -2, 0])

a31 = 0.7
a12 = 0.2
a21 = 0.2
a32 = 0.7

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

# Simuation Time
t = np.arange(0, 20, 0.1)
# Simulationa Initial
t1_lap_initial, y1_lap_initial, x1_lap_initial = control.initial_response(
    sys_Laplace, T=t, X0=X0_initial_m2, return_x=True
)
# Simulation Impulse
t2_lap_initial, y2_lap_initial, x2_lap_initial = control.initial_response(
    sys_Laplace, T=t, X0=X0_initial_m4, return_x=True
)


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

# Mark the last values on the plot
last_value_index = -1
last_value_A = x1_lap_initial[0, last_value_index]
last_value_B = x1_lap_initial[1, last_value_index]

plt.scatter(t1_lap_initial[last_value_index], last_value_A, color='red', marker='o')
plt.scatter(t1_lap_initial[last_value_index], last_value_B, color='blue', marker='o')

# Display the last values
plt.text(t1_lap_initial[last_value_index], last_value_A, f'A: {last_value_A:.2f}', ha='right', va='bottom')
plt.text(t1_lap_initial[last_value_index], last_value_B, f'B: {last_value_B:.2f}', ha='right', va='top')

# # Figures to Simulation Impulse
plt.figure()
plt.title("Resposta a condicao Inicial Momento 4 para o 5")
plt.plot(t2_lap_initial, x2_lap_initial[0, :], label='Agent A')
plt.plot(t2_lap_initial, x2_lap_initial[1, :], label='Agent B')
plt.xlabel("tempo (s)")
plt.ylabel("Posição")
plt.legend(["Agente A", "Agente B"])
plt.grid()

# Define the time array
t = np.arange(0, 20, 0.1)

# Define the input signal F for a step input with value 2
F = np.ones_like(t) * 1 # Step input with value 2

# Simulate the system response to the forced input
t_forced, y_forced, x_forced = control.forced_response(sys_Laplace, t, F, X0_initial_m2,return_x=True)

# Extract individual states
x1 = x_forced[0, :]
x2 = x_forced[1, :]
x3 = x_forced[2, :]

# Plot the forced response
plt.figure(figsize=(30, 6))

# Plot x1, x2, and y on the same graph
plt.plot(t_forced, x1, label='Agent A')
plt.plot(t_forced, x2, label='Agent B')
#plt.plot(t_forced, x3, label='Leader')

# Add labels and legend
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()

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
plt.plot(t_forced, x1, label='Agent A (Position)')
plt.plot(t_forced, v1, label='Agent A (Velocity)', linestyle='-.', color='blue')

plt.plot(t_forced, x2, label='Agent B (Position)')
plt.plot(t_forced, v2, label='Agent B (Velocity)', linestyle='-.', color='orange')

# plt.plot(t_forced, x3, label='Leader (Position)')
plt.plot(t_forced, v3, label='Leader (Velocity)', linestyle='-.', color='red')

# Mark the last values on the plot
last_value_index = -1
last_value_A = x_forced[0, last_value_index]
last_value_B = x_forced[1, last_value_index]
last_value_C = v3[last_value_index]

plt.scatter(t_forced[last_value_index], last_value_A, color='red', marker='o')
plt.scatter(t_forced[last_value_index], last_value_B, color='blue', marker='o')
plt.scatter(t_forced[last_value_index], last_value_C, color='green', marker='o')

# Display the last values
plt.text(t_forced[last_value_index], last_value_A, f'A: {last_value_A:.2f}', ha='right', va='bottom')
plt.text(t_forced[last_value_index], last_value_B, f'B: {last_value_B:.2f}', ha='right', va='top')
plt.text(t_forced[last_value_index], last_value_C, f'C: {last_value_C:.2f}', ha='right', va='top')


# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Position / Velocity')
plt.legend()

# Add grid
plt.grid()
plt.show()


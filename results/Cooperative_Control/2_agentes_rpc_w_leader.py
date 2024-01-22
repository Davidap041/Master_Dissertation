import control
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Double Integrator System for 2 agents
# Moment 1
# Moment 2
X0_initial_m2 = np.array([2, -1, 0, 0, 0, 0])
X0_initial_m4 = np.array([-2, -2, 0, 0, 0, 0])

a31 = 0.5
a12 = 0.1
a21 = 0.1
a32 = 0.5

# Define the state-space matrices for the double integrator
A_double_integrator = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

B_double_integrator = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])

C_double_integrator = np.eye(6)
D_double_integrator = np.zeros((6, 2))

# Create the state-space system
sys_double_integrator = control.ss(A_double_integrator, B_double_integrator, C_double_integrator, D_double_integrator)

# Simuation Time
t = np.arange(0, 20, 0.1)
# Simulation Initial
t1_double_integrator_initial, y1_double_integrator_initial, x1_double_integrator_initial = control.initial_response(
    sys_double_integrator, T=t, X0=X0_initial_m2, return_x=True
)
# Simulation Impulse
t2_double_integrator_initial, y2_double_integrator_initial, x2_double_integrator_initial = control.initial_response(
    sys_double_integrator, T=t, X0=X0_initial_m4, return_x=True
)

# Figures to Simulation Initial
plt.figure()
plt.title("Response to Initial Condition - Moment 2 to 3")
plt.plot(t1_double_integrator_initial, x1_double_integrator_initial[0, :], label='Agent A')
plt.plot(t1_double_integrator_initial, x1_double_integrator_initial[1, :], label='Agent B')
plt.plot(t1_double_integrator_initial, x1_double_integrator_initial[4, :], label='Leader A')
plt.plot(t1_double_integrator_initial, x1_double_integrator_initial[5, :], label='Leader B')
plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity")
plt.legend()
plt.grid()

# Figures to Simulation Impulse
plt.figure()
plt.title("Response to Initial Condition - Moment 4 to 5")
plt.plot(t2_double_integrator_initial, x2_double_integrator_initial[0, :], label='Agent A')
plt.plot(t2_double_integrator_initial, x2_double_integrator_initial[1, :], label='Agent B')
plt.plot(t2_double_integrator_initial, x2_double_integrator_initial[4, :], label='Leader A')
plt.plot(t2_double_integrator_initial, x2_double_integrator_initial[5, :], label='Leader B')
plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity")
plt.legend()
plt.grid()


# Define the time array
t = np.arange(0, 20, 0.1)

# Define the input signal F for a step input with value 2
F = np.ones_like(t) * 0.2  # Step input with value 2

# Simulate the system response to the forced input
t_forced, y_forced, x_forced = control.forced_response(sys_double_integrator, t, F, X0_initial_m2, return_x=True)

# Extract individual states
x1 = x_forced[0, :]
x2 = x_forced[1, :]
y = y_forced[0, :]  # Assuming you want the first output (position) for y

# Plot the forced response
plt.figure(figsize=(30, 6))

# Plot x1, x2, and y on the same graph
plt.plot(t_forced, x1, label='Agent A')
plt.plot(t_forced, x2, label='Agent B')
plt.plot(t_forced, y, label='Leader Position', linestyle='--', color='black')

# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()

# Simulate the system response to the forced input
t_forced, y_forced, x_forced = control.forced_response(sys_double_integrator, t, F, X0_initial_m2, return_x=True)

# Extract individual states
x1 = x_forced[0, :]
x2 = x_forced[1, :]

# Calculate velocity
dt = t_forced[1] - t_forced[0]
v1 = np.gradient(x1, dt)
v2 = np.gradient(x2, dt)

# Plot the forced response with velocity
plt.figure(figsize=(10, 6))

# Plot position and velocity on the same graph
plt.plot(t_forced, x1, label='Agent A (Position)')
plt.plot(t_forced, v1, label='Agent A (Velocity)', linestyle='-.', color='blue')

plt.plot(t_forced, x2, label='Agent B (Position)')
plt.plot(t_forced, v2, label='Agent B (Velocity)', linestyle='-.', color='orange')

plt.plot(t_forced, y, label='Leader Position', linestyle='--', color='black')

# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Position / Velocity')
plt.legend()

# Add grid
plt.grid()
plt.show()


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

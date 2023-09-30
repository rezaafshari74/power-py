import numpy as np

# Load grid parameters
# Assuming NE_test_parameters.py contains the necessary parameters
from NE_test_parameters import *

# Initial stable state
y0 = np.array([0.1564, 0.1806, 0.1631, 0.3135, 0.1823, 0.1849, 0.1652, 0.2953, -0.06165, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y0 = np.zeros(18)  # Alternatively, initialize with zeros

# Sampling time
delta_t = 2.5e-4
# Final simulation time
T_sim = 15
# Vector of simulation times
tspan = np.arange(0, T_sim + delta_t, delta_t)
# Fault initial time
T_fault_in = 2
# Fault final time
T_fault_end = 2.525
# Number of inputs
m = 9
# Number of states
n = 18
# Control horizon
T = 0.1
# Final stable state
xf = y0
# Control starting time
t_c = 3

k = 1  # You can continue with your calculations from here
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the swing equations
def swing(t, delta_t, y, u):
    # Update only the first 9 states
    y[:9] += u
    return y


# Simulation parameters
tspan = np.arange(0, T_sim + delta_t, delta_t)
yy = np.zeros((n, len(tspan)))
k = 0

# Simulate fault (discretized swing equations)
for t in tspan:
    yy[:, k] = swing(t, delta_t, y0, np.zeros(m))
    y0 = yy[:, k]
    k += 1

# Initial state control
x0 = yy[:, int(round(t_c / delta_t))]

# Plot results
plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(tspan, yy[0:9, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 200, 200, -20], color='r', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.subplot(2, 2, 3)
plt.plot(tspan, yy[9:18, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 60, 60, -20], color='r', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.show()
import numpy as np

# Function to simulate the swing equations
def swing(t, delta_t, y, u):
    # Replace this function with your swing equations implementation
    # The function should return the updated state vector
    # For demonstration purposes, we'll assume it updates 'y' by adding 'u' to it
    return y + u

# Simulation parameters
N = 5000  # Number of samples
sigma = 0.01  # Variance of initial states

# Data matrices
U = np.empty((m * round(T / delta_t), 0))
X0 = np.empty((n, 0))
X1T = np.empty((n * (round(T / delta_t) - 1), 0))
XT = np.empty((n, 0))

for i in range(N):
    k = 1
    
    # Random input
    u = 1e-3 * np.random.randn(m, round(T / delta_t))
    # Random initial state
    y0 = np.array([0.1564 + sigma * np.random.randn(),
                   0.1806 + sigma * np.random.randn(),
                   0.1631 + sigma * np.random.randn(),
                   0.3135 + sigma * np.random.randn(),
                   0.1823 + sigma * np.random.randn(),
                   0.1849 + sigma * np.random.randn(),
                   0.1652 + sigma * np.random.randn(),
                   0.2953 + sigma * np.random.randn(),
                  -0.06165 + sigma * np.random.randn()] + [sigma * np.random.randn() for _ in range(9)])
    
    y0_data = y0.copy()
    y_data = np.empty((n, round(T / delta_t)))

    for t in np.arange(0, T - delta_t, delta_t):
        y_data[:, k - 1] = swing(t, delta_t, y0_data, u[:, k - 1])
        y0_data = y_data[:, k - 1]
        k += 1
    
    U = np.concatenate((U, np.fliplr(u).reshape(-1, 1)), axis=1)
    X0 = np.concatenate((X0, y0.reshape(-1, 1)), axis=1)
    X1T = np.concatenate((X1T, np.fliplr(y_data[:, :round(T / delta_t) - 1]).reshape(-1, 1)), axis=1)
    XT = np.concatenate((XT, y_data[:, -1].reshape(-1, 1)), axis=1)
import numpy as np
from scipy.linalg import pinv, null, svd, cholesky

# Define the null space tolerance
tolerance = 1e-10

# Keep a subset of the data
U = U[:, :3750]
XT = XT[:, :3750]
X0 = X0[:, :3750]
X1T = X1T[:, :3750]

# Calculate null spaces
K_X0 = null(X0, rcond=tolerance)
K_U = null(U, rcond=tolerance)

# Compute xf_c
xf_c = xf - np.dot(XT, K_U @ pinv(K_U.T @ X0.T, rcond=tolerance)) @ x0

# Project data onto K_X0 subspace
U = U @ K_X0
X1T = X1T @ K_X0
XT = XT @ K_X0

# Recalculate null spaces
K_U = null(U, rcond=tolerance)
K_XT = null(XT, rcond=tolerance)

# Define control cost parameters
Q = 5e-3
R = 1

# Compute L
L2 = Q * (X1T.T @ X1T) + R * (U.T @ U)
L = cholesky(L2, lower=True)

# Compute SVD
U_svd, S_svd, V_svd = svd(L @ K_XT, full_matrices=False)
W = U @ pinv(XT, rcond=tolerance) @ xf_c - U @ K_XT @ pinv(W @ S_svd @ V_svd.T, rcond=tolerance) @ L @ pinv(XT, rcond=tolerance) @ xf_c

# Reshape the optimal control input
u_opt_seq = np.fliplr(u_opt_seq.reshape((m, round(T / delta_t))))

# Apply control for fault recovery
u_opt_seq = np.hstack((np.zeros((9, round(t_c / delta_t) + 1)), u_opt_seq, np.zeros((9, round(500 / delta_t)))))

y0 = np.zeros((18,))  # Initial state
yy = np.zeros((18, len(tspan)))

for k, t in enumerate(tspan):
    yy[:, k] = swing(t, delta_t, y0, u_opt_seq[:, k])
    y0 = yy[:, k]

# The 'yy' variable now contains the state trajectories over time after applying control
import matplotlib.pyplot as plt

# Define the time vector for plotting
t_sim = np.arange(0, T_sim + delta_t, delta_t)

# Define the time ranges for fault and control
fault_range = [T_fault_in, T_fault_end]
control_range = [t_c, t_c + T]

# Plot the results
plt.figure(1, figsize=(10, 8))

# Phase plot
plt.subplot(2, 2, 2)
for i in range(9):
    plt.plot(t_sim, yy[i, :round(T_sim / delta_t) + 1], label=f'gen {i + 1}')
plt.fill_between(fault_range, -10, 15, color='red', alpha=0.25)
plt.fill_between(control_range, -10, 15, color='green', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(loc='upper right')

# Frequency plot
plt.subplot(2, 2, 4)
for i in range(9):
    plt.plot(t_sim, yy[i + 9, :round(T_sim / delta_t) + 1], label=f'gen {i + 1}')
plt.fill_between(fault_range, -10, 15, color='red', alpha=0.25)
plt.fill_between(control_range, -10, 15, color='green', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(loc='upper right')

# Show the plots
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Define the time vector for plotting
tspan = np.arange(0, T_sim + delta_t, delta_t)

# Define the time range for fault and control
fault_range = [T_fault_in, T_fault_end]
control_range = [t_c, t_c + T]

# Plot the asymptotic behavior
plt.figure(2, figsize=(10, 8))

# Phase plot
plt.subplot(2, 1, 1)
for i in range(9):
    plt.plot(tspan, yy[i, :round(T_sim / delta_t) + 1], label=f'gen {i + 1}')
plt.fill_between([2, 2.4], -10, 15, color='red', alpha=0.25)
plt.fill_between(control_range, -10, 15, color='green', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(loc='upper right')

# Frequency plot
plt.subplot(2, 1, 2)
for i in range(9):
    plt.plot(tspan, yy[i + 9, :round(T_sim / delta_t) + 1], label=f'gen {i + 1}')
plt.fill_between([2, 2.4], -10, 15, color='red', alpha=0.25)
plt.fill_between(control_range, -10, 15, color='green', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(loc='upper right')

# Show the plots
plt.tight_layout()
plt.show()

# Define the swing function
def swing(t, delta_t, y, u):
    # Implement your swing equation here
    # Replace this with your actual code for the swing equation
    pass
import numpy as np

def swing(t, delta_t, y, u):
    global H, P, D, E, Y3, Y32, T_fault_in, T_fault_end

    yp = np.zeros(18)
    
    Y = Y3
    
    k = np.array([
        delta_t * (1 / H[1, 1]) * (P[1] - np.real(Y[1, 1]) * E[1] ** 2 - E[1] * E[0] * np.real(Y[1, 0]) - E[1] * E[2] * np.real(Y[1, 2]) - E[1] * E[3] * (np.real(Y[1, 3])) - E[1] * E[4] * (np.real(Y[1, 4])) - E[1] * E[5] * (np.real(Y[1, 5])) - E[1] * E[6] * (np.real(Y[1, 6])) - E[1] * E[7] * (np.real(Y[1, 7])) - E[1] * E[8] * (np.real(Y[1, 8])) - E[1] * E[9] * (np.real(Y[1, 9])) - E[1] * E[10] * (np.real(Y[1, 10]))),
        delta_t * (1 / H[2, 2]) * (P[2] - np.real(Y[2, 2]) * E[2] ** 2 - E[2] * E[0] * (np.real(Y[2, 0])) - E[2] * E[1] * (np.real(Y[2, 1])) - E[2] * E[3] * (np.real(Y[2, 3])) - E[2] * E[4] * (np.real(Y[2, 4])) - E[2] * E[5] * (np.real(Y[2, 5])) - E[2] * E[6] * (np.real(Y[2, 6])) - E[2] * E[7] * (np.real(Y[2, 7])) - E[2] * E[8] * (np.real(Y[2, 8])) - E[2] * E[9] * (np.real(Y[2, 9])) - E[2] * E[10] * (np.real(Y[2, 10]))),
        delta_t * (1 / H[3, 3]) * (P[3] - np.real(Y[3, 3]) * E[3] ** 2 - E[3] * E[0] * (np.real(Y[3, 0])) - E[3] * E[1] * (np.real(Y[3, 1])) - E[3] * E[2] * (np.real(Y[3, 2])) - E[3] * E[4] * (np.real(Y[3, 4])) - E[3] * E[5] * (np.real(Y[3, 5])) - E[3] * E[6] * (np.real(Y[3, 6])) - E[3] * E[7] * (np.real(Y[3, 7])) - E[3] * E[8] * (np.real(Y[3, 8])) - E[3] * E[9] * (np.real(Y[3, 9])) - E[3] * E[10] * (np.real(Y[3, 10]))),
        delta_t * (1 / H[4, 4]) * (P[4] - np.real(Y[4, 4]) * E[4] ** 2 - E[4] * E[0] * (np.real(Y[4, 0])) - E[4] * E[1] * (np.real(Y[4, 1])) - E[4] * E[2] * (np.real(Y[4, 2])) - E[4] * E[3] * (np.real(Y[4, 3])) - E[4] * E[5] * (np.real(Y[4, 5])) - E[4] * E[6] * (np.real(Y[4, 6])) - E[4] * E[7] * (np.real(Y[4, 7])) - E[4] * E[8] * (np.real(Y[4, 8])) - E[4] * E[9] * (np.real(Y[4, 9])) - E[4] * E[10] * (np.real(Y[4, 10]))),
        delta_t * (1 / H[5, 5]) * (P[5] - np.real(Y[5, 5]) * E[5] ** 2 - E[5] * E[0] * (np.real(Y[5, 0])) - E[5] * E[1] * (np.real(Y[5, 1])) - E[5] * E[2] * (np.real(Y[5, 2])) - E[5] * E[3] * (np.real(Y[5, 3])) - E[5] * E[4] * (np.real(Y[5, 4])) - E[5] * E[6] * (np.real(Y[5, 6])) - E[5] * E[7] * (np.real(Y[5, 7])) - E[5] * E[8] * (np.real(Y[5, 8])) - E[5] * E[9] * (np.real(Y[5, 9])) - E[5] * E[10] * (np.real(Y[5, 10]))),
        delta_t * (1 / H[6, 6]) * (P[6] - np.real(Y[6, 6]) * E[6] ** 2 - E[6] * E[0] * (np.real(Y[6, 0])) - E[6] * E[1] * (np.real(Y[6, 1])) - E[6] * E[2] * (np.real(Y[6, 2])) - E[6] * E[3] * (np.real(Y[6, 3])) - E[6] * E[4] * (np.real(Y[6, 4])) - E[6] * E[5] * (np.real(Y[6, 5])) - E[6] * E[7] * (np.real(Y[6, 7])) - E[6] * E[8] * (np.real(Y[6, 8])) - E[6] * E[9] * (np.real(Y[6, 9])) - E[6] * E[10] * (np.real(Y[6, 10]))),
        delta_t * (1 / H[7, 7]) * (P[7] - np.real(Y[7, 7]) * E[7] ** 2 - E[7] * E[0] * (np.real(Y[7, 0])) - E[7] * E[1] * (np.real(Y[7, 1])) - E[7] * E[2] * (np.real(Y[7, 2])) - E[7] * E[3] * (np.real(Y[7, 3])) - E[7] * E[4] * (np.real(Y[7, 4])) - E[7] * E[5] * (np.real(Y[7, 5])) - E[7] * E[6] * (np.real(Y[7, 6])) - E[7] * E[8] * (np.real(Y[7, 8])) - E[7] * E[9] * (np.real(Y[7, 9])) - E[7] * E[10] * (np.real(Y[7, 10]))),
        delta_t * (1 / H[8, 8]) * (P[8] - np.real(Y[8, 8]) * E[8] ** 2 - E[8] * E[0] * (np.real(Y[8, 0])) - E[8] * E[1] * (np.real(Y[8, 1])) - E[8] * E[2] * (np.real(Y[8, 2])) - E[8] * E[3] * (np.real(Y[8, 3])) - E[8] * E[4] * (np.real(Y[8, 4])) - E[8] * E[5] * (np.real(Y[8, 5])) - E[8] * E[6] * (np.real(Y[8, 6])) - E[8] * E[7] * (np.real(Y[8, 7])) - E[8] * E[9] * (np.real(Y[8, 9])) - E[8] * E[10] * (np.real(Y[8, 10]))),
        delta_t * (1 / H[9, 9]) * (P[9] - np.real(Y[9, 9]) * E[9] ** 2 - E[9] * E[0] * (np.real(Y[9, 0])) - E[9] * E[1] * (np.real(Y[9, 1])) - E[9] * E[2] * (np.real(Y[9, 2])) - E[9] * E[3] * (np.real(Y[9, 3])) - E[9] * E[4] * (np.real(Y[9, 4])) - E[9] * E[5] * (np.real(Y[9, 5])) - E[9] * E[6] * (np.real(Y[9, 6])) - E[9] * E[7] * (np.real(Y[9, 7])) - E[9] * E[8] * (np.real(Y[9, 8])) - E[9] * E[10] * (np.real(Y[9, 10]))),
        delta_t * (1 / H[10, 10]) * (P[10] - np.real(Y[10, 10]) * E[10] ** 2 - E[10] * E[0] * (np.real(Y[10, 0])) - E[10] * E[1] * (np.real(Y[10, 1])) - E[10] * E[2] * (np.real(Y[10, 2])) - E[10] * E[3] * (np.real(Y[10, 3])) - E[10] * E[4] * (np.real(Y[10, 4])) - E[10] * E[5] * (np.real(Y[10, 5])) - E[10] * E[6] * (np.real(Y[10, 6])) - E[10] * E[7] * (np.real(Y[10, 7])) - E[10] * E[8] * (np.real(Y[10, 8])) - E[10] * E[9] * (np.real(Y[10, 9]))),
    ])
    
    if t < 2:
        Y = Y3
    elif T_fault_in <= t < T_fault_end:
        Y = Y32
    else:
        Y = Y3
    
    return yp
import numpy as np

def calculate_yp(y, u, delta_t, H, D, P, Y, E, k):
    yp = np.zeros(11)
    
    for i in range(10):
        yp[i] = delta_t * y[i + 9] + y[i] + u[i]
    
    yp[10] = delta_t * (1 / H[1, 1]) * (-D[1, 1] * (y[9] + u[0]) + P[1] - np.real(Y[1, 1]) * E[1]**2 - 
        E[1] * E[0] * (np.real(Y[1, 0]) * np.cos(y[0]) + np.imag(Y[1, 0]) * np.sin(y[0])) - 
        np.sum(E[1] * E[j] * (np.real(Y[1, j]) * np.cos(y[j - 1]) + np.imag(Y[1, j]) * np.sin(y[j - 1])) 
                 for j in range(2, 11)) 
        ) + y[9] - k[0]
    
    return yp

# Example usage:
y = np.array([1.0] * 18)
u = np.array([0.0] * 9)
delta_t = 0.1
H = np.eye(10)
D = np.eye(10)
P = np.array([0.0] * 10)
Y = np.eye(10)
E = np.array([1.0] * 10)
k = np.array([0.0] * 9)

yp_result = calculate_yp(y, u, delta_t, H, D, P, Y, E, k)
print(yp_result)


yp[11] = delta_t * (1 / H[2, 2]) * (-D[2, 2] * (y[10] + u[0]) + P[2] - np.real(Y[2, 2]) * E[2]**2 -
        E[2] * E[0] * (np.real(Y[2, 0]) * np.cos(y[0]) + np.imag(Y[2, 0]) * np.sin(y[0])) - 
        np.sum(E[2] * E[j] * (np.real(Y[2, j]) * np.cos(y[j - 1]) + np.imag(Y[2, j]) * np.sin(y[j - 1])) 
                for j in range(3, 11))
        ) + y[10] - k[1]


yp[13] = delta_t * (1 / H[4, 4]) * (-D[4, 4] * (y[12] + u[2]) + P[4] - np.real(Y[4, 4]) * E[4]**2 -
        E[4] * E[0] * (np.real(Y[4, 0]) * np.cos(y[2]) + np.imag(Y[4, 0]) * np.sin(y[2])) -
        np.sum(E[4] * E[j] * (np.real(Y[4, j]) * np.cos(y[j - 1]) + np.imag(Y[4, j]) * np.sin(y[j - 1])) 
                for j in range(3, 11))
        ) + y[12] - k[2]


# ... (continue with the rest of the yp equations)

# Example usage:
y = np.array([1.0] * 18)
u = np.array([0.0] * 9)
delta_t = 0.1
H = np.eye(10)
D = np.eye(10)
P = np.array([0.0] * 10)
Y = np.eye(10)
E = np.array([1.0] * 10)
k = np.array([0.0] * 9)

yp_result = calculate_yp(y, u, delta_t, H, D, P, Y, E, k)
print(yp_result)


yp[14] = delta_t * (1 / H[5, 5]) * (-D[5, 5] * (y[13] + u[4]) + P[5] - np.real(Y[5, 5]) * E[5]**2 - 
        E[5] * E[0] * (np.real(Y[5, 0]) * np.cos(y[3]) + np.imag(Y[5, 0]) * np.sin(y[3])) - 
        np.sum(E[5] * E[j] * (np.real(Y[5, j]) * np.cos(y[j - 1]) + np.imag(Y[5, j]) * np.sin(y[j - 1])) 
                 for j in range(4, 11)) 
        ) + y[13] - k[4]

yp[15] = delta_t * (1 / H[6, 6]) * (-D[6, 6] * (y[14] + u[5]) + P[6] - np.real(Y[6, 6]) * E[6]**2 - 
        E[6] * E[0] * (np.real(Y[6, 0]) * np.cos(y[4]) + np.imag(Y[6, 0]) * np.sin(y[4])) - 
        np.sum(E[6] * E[j] * (np.real(Y[6, j]) * np.cos(y[j - 1]) + np.imag(Y[6, j]) * np.sin(y[j - 1])) 
                 for j in range(4, 11)) 
        ) + y[14] - k[5]

yp[16] = delta_t * (1 / H[7, 7]) * (-D[7, 7] * (y[15] + u[6]) + P[7] - np.real(Y[7, 7]) * E[7]**2 - 
        E[7] * E[0] * (np.real(Y[7, 0]) * np.cos(y[5]) + np.imag(Y[7, 0]) * np.sin(y[5])) - 
        np.sum(E[7] * E[j] * (np.real(Y[7, j]) * np.cos(y[j - 1]) + np.imag(Y[7, j]) * np.sin(y[j - 1])) 
                 for j in range(4, 11)) 
        ) + y[15] - k[6]

yp[17] = delta_t * (1 / H[8, 8]) * (-D[8, 8] * (y[16] + u[7]) + P[8] - np.real(Y[8, 8]) * E[8]**2 - 
        E[8] * E[0] * (np.real(Y[8, 0]) * np.cos(y[6]) + np.imag(Y[8, 0]) * np.sin(y[6])) - 
        np.sum(E[8] * E[j] * (np.real(Y[8, j]) * np.cos(y[j - 1]) + np.imag(Y[8, j]) * np.sin(y[j - 1])) 
                 for j in range(4, 11)) 
        ) + y[16] - k[7]

# ... (continue with the rest of the yp equations)

# Example usage:
y = np.array([1.0] * 18)
u = np.array([0.0] * 9)
delta_t = 0.1
H = np.eye(10)
D = np.eye(10)
P = np.array([0.0] * 10)
Y = np.eye(10)
E = np.array([1.0] * 10)
k = np.array([0.0] * 9)

yp_result = calculate_yp(y, u, delta_t, H, D, P, Y, E, k)
print(yp_result)

yp[18] = delta_t * (1 / H[9, 9]) * (-D[9, 9] * (y[17] + u[8]) + P[9] - np.real(Y[9, 9]) * E[9]**2 - 
    np.sum(E[9] * E[j] * (np.real(Y[9, j]) * np.cos(y[j - 1]) + np.imag(Y[9, j]) * np.sin(y[j - 1])) 
             for j in range(1, 11)) 
    ) + y[17] - k[8]

yp[19] = delta_t * (1 / H[10, 10]) * (-D[10, 10] * (y[18] + u[9]) + P[10] - np.real(Y[10, 10]) * E[10]**2 - 
    np.sum(E[10] * E[j] * (np.real(Y[10, j]) * np.cos(y[j - 1]) + np.imag(Y[10, j]) * np.sin(y[j - 1])) 
             for j in range(1, 11)) 
    ) + y[18] - k[9]

# ... (continue with the rest of the yp equations)

# Example usage:
y = np.array([1.0] * 18)
u = np.array([0.0] * 9)
delta_t = 0.1
H = np.eye(10)
D = np.eye(10)
P = np.array([0.0] * 10)
Y = np.eye(10)
E = np.array([1.0] * 10)
k = np.array([0.0] * 9)

yp_result = calculate_yp(y, u, delta_t, H, D, P, Y, E, k)
print(yp_result)

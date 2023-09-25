import numpy as np
import matplotlib.pyplot as plt
from swing_simulation import swing  # Import the swing function from swing_simulation

# Load grid parameters from NE_test_parameters
from NE_test_parameters import *

# Set initial stable state
y0 = np.array([0.1564, 0.1806, 0.1631, 0.3135, 0.1823, 0.1849, 0.1652, 0.2953, -0.06165, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y0 = np.zeros(18)  # Alternatively, you can initialize all elements to zero

# Set simulation parameters
delta_t = 2.5e-4
T_sim = 15
tspan = np.arange(0, T_sim + delta_t, delta_t)
T_fault_in = 2
T_fault_end = 2.525
m = 9
n = 18
T = 0.1
xf = y0.copy()
t_c = 3
k = 1

# Simulate fault (discretized swing equations)
yy = np.zeros((n, len(tspan)))

for t in tspan:
    yy[:, k - 1] = swing(t, delta_t, y0, np.zeros(m))
    y0 = yy[:, k - 1].copy()
    k += 1

# Initial state control
x0 = yy[:, int(round(t_c / delta_t))]

# Plot results
plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(tspan, yy[0:9, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 200, 200, -20], 'r', alpha=0.25, linestyle='none')
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])
plt.subplot(2, 2, 3)
plt.plot(tspan, yy[9:18, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 60, 60, -20], 'r', alpha=0.25, linestyle='none')
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

# Generate data for control
N = 5000
sigma = 0.01
U = np.zeros((m * round(T / delta_t), N))
X0 = np.zeros((n, N))
X1T = np.zeros((n * (round(T / delta_t) - 1), N))
XT = np.zeros((n, N))

for i in range(N):
    k = 1
    u = 1e-3 * np.random.randn(m, int(round(T / delta_t)))
    y0 = np.array([0.1564 + sigma * np.random.randn(),
                   0.1806 + sigma * np.random.randn(),
                   0.1631 + sigma * np.random.randn(),
                   0.3135 + sigma * np.random.randn(),
                   0.1823 + sigma * np.random.randn(),
                   0.1849 + sigma * np.random.randn(),
                   0.1652 + sigma * np.random.randn(),
                   0.2953 + sigma * np.random.randn(),
                   -0.06165 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn(),
                   0 + sigma * np.random.randn()])
    
    y0_data = y0.copy()
    
    for t in np.arange(0, T - delta_t, delta_t):
        y_data = swing(t, delta_t, y0_data, u[:, k - 1])
        y0_data = y_data.copy()
        k += 1
    
    U[:, i] = np.flipud(u).reshape(-1)
    X0[:, i] = y0
    X1T[:, i] = np.flipud(y_data - xf).reshape(-1)
    XT[:, i] = y_data

# Save data
np.savez('data_m2_x0_5000', U=U, X0=X0, X1T=X1T, XT=XT)

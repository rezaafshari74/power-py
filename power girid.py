import numpy as np
import matplotlib.pyplot as plt

# Define global variables
global H, P, D, E, Y3, Y32, T_fault_in, T_fault_end

# Load grid parameters (assuming NE_test_parameters is a separate file)
from NE_test_parameters import *

# Initial stable state
y0 = np.array([0.1564, 0.1806, 0.1631, 0.3135, 0.1823, 0.1849, 0.1652, 0.2953, -0.06165, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Alternatively, for the second y0 assignment:
# y0 = np.zeros(18)

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

k = 1

# Simulate fault (discretized swing equations)
import numpy as np
import matplotlib.pyplot as plt

# Simulate fault (discretized swing equations)
yy = np.zeros((n, len(tspan)))

# Assuming tspan is a 1D array with 60001 elements
tspan = np.linspace(0, 200, 60001)  # You should adjust the start and end values accordingly

# Assuming yy is a 2D array with shape (10, 60001)

# Continue with your simulation and calculation of yy here...

# After you have calculated yy, you can now plot it
# Plot the first 9 rows of yy against tspan for the same number of time points
plt.figure()
for i in range(9):
    plt.plot(tspan[:len(yy[i, :])], yy[i, :])

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Plot of First 9 Rows of yy')
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
def swing(t, delta_t, y0, control_input):
    # Implement your swing function here
    pass
import numpy as np

# Initial array with 60000 columns
yy = np.zeros((n, 60000))

# Check if k exceeds the current array size
if k > yy.shape[1]:
    # Resize the array to accommodate more columns
    new_columns = k - yy.shape[1] + 1
    yy = np.hstack((yy, np.zeros((n, new_columns))))

# Now you can safely assign values to yy[:, k - 1]
yy[:, k - 1] = swing(t, delta_t, y0, np.zeros(m))

for t in tspan:
    yy[:, k - 1] = swing(t, delta_t, y0, np.zeros(m))
    y0 = yy[:, k - 1]
    k += 1
import numpy as np
import matplotlib.pyplot as plt

# Define your variables (yy, final_time, num_time_steps, num_variables, delta_t, m) here
import numpy as np
import matplotlib.pyplot as plt

# Define your variables (yy, final_time, num_time_steps, num_variables, delta_t, m) here
import numpy as np
import matplotlib.pyplot as plt

# Define your variables (yy, final_time, num_time_steps, num_variables, delta_t, m) here
final_time = 10  # Replace with the actual final time
num_time_steps = 100  # Adjust this number to match the number of time steps used in your data generation

# Define the initial conditions and other necessary variables (yy, delta_t, m) here
# yy = ...
# delta_t = ...
# m = ...

tspan = np.linspace(0, final_time, num_time_steps)

num_variables = 18  # Set this to match the number of columns in yy

# Initialize k, yy, and t
k = 1
yy = np.zeros((num_variables, 1))  # Start with a single column
t = 0  # Initialize t here

# Now you can use tspan in your loop
# Define 'T' with an appropriate value
T = 0  # Assign a suitable initial value to T

# Now you can use 'T' in your loop
# Loop through 'tspan' directly
for t in tspan:
    # Check if k exceeds the current array size
    if k > yy.shape[1]:
        # Resize the array to accommodate more columns
        new_columns = k - yy.shape[1] + 1
        yy = np.hstack((yy, np.zeros((num_variables, new_columns))))

    # Use 't' from 'tspan' directly
    yy[:, k - 1] = swing(t, delta_t, y0, np.zeros(m))

    y0 = yy[:, k - 1]
    k += 1






# You can also plot your results or perform further analysis here


# You can also plot your results or perform further analysis here


# You can also plot your results or perform further analysis here


# Assuming you want to plot all data points
# Remove the subset selection
subset_of_yy = yy[:, :10]  # Select the first 10 columns of yy


# Adjust tspan to have the same number of elements as variables (18 in this case)
tspan = np.linspace(0, final_time, num_variables)
for i in range(subset_of_yy.shape[1]):
    plt.plot(tspan, subset_of_yy[:, i])
  # Adjust the range and number of points as needed

# Now you can plot the data for all variables
for i in range(yy.shape[1]):  # Use yy instead of subset_of_yy
    plt.plot(tspan, yy[:, i])

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Your Title Here')
plt.show()
import matplotlib.pyplot as plt

# Ensure tspan has the same length as the number of data points in yy
tspan = your_tspan_array  # Replace with your actual tspan data

# Iterate through the columns of yy and plot each one
for i in range(yy.shape[1]):
    plt.plot(tspan, yy[:, i])

# Add labels, legend, and show the plot
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(['Curve 1', 'Curve 2', 'Curve 3', ...])  # Add labels for each curve
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define your variables (yy, final_time, num_time_steps, num_variables) here
final_time = 10  # Replace with the actual final time
num_time_steps = 100  # Adjust this number to match the number of time steps used in your data generation



# Create an array of time values matching the number of variables
tspan = np.linspace(0, final_time, num_variables)

# Assuming you have your data in the variable yy
# yy = ...  # Replace with your data

# Now you can plot the data for all variables
for i in range(yy.shape[1]):
    plt.plot(tspan, yy[:, i])

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Your Title Here')
plt.show()



# Assuming yy is a numpy array with shape (num_variables, num_time_steps)
# where num_variables is the number of variables you want to plot
# and num_time_steps is the number of time steps
# Example:
# yy = np.array([[...], [...], ...])

# Define tspan with the correct number of time steps
tspan = np.linspace(0, final_time, num_time_steps)

# Then, you can plot the variables
for i in range(num_variables):
    plt.plot(tspan, yy[i, :])

plt.xlabel('Time')
plt.ylabel('Variable Value')
plt.title('Variable vs. Time')
plt.show()

# Rest of your main code
# ...


# Initial state control
x0 = yy[:, int(round(t_c / delta_t))]

# Plot results
plt.figure(1)

plt.subplot(2, 2, 1)
plt.plot(tspan, yy[0:9, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 200, 200, -20], 'r', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.subplot(2, 2, 3)
plt.plot(tspan, yy[9:18, :])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-20, 60, 60, -20], 'r', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.show()
import numpy as np
from scipy.linalg import cholesky

# Number of samples
N = 5000

# Variance of initial states
sigma = 0.01

# Data matrices
U = []
X0 = []
X1T = []
XT = []

for i in range(N):
    k = 1
    
    # Random input
    u = 1e-3 * np.random.randn(m, int(round(T / delta_t)))
    
    # Random initial state
    y0 = [0.1564 + sigma * np.random.randn(),
          0.1806 + sigma * np.random.randn(),
          0.1631 + sigma * np.random.randn(),
          0.3135 + sigma * np.random.randn(),
          0.1823 + sigma * np.random.randn(),
          0.1849 + sigma * np.random.randn(),
          0.1652 + sigma * np.random.randn(),
          0.2953 + sigma * np.random.randn(),
         -0.06165 + sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn(),
          sigma * np.random.randn()]
    
    y0_data = y0.copy()
    
    y_data = np.zeros((n, int(round(T / delta_t))))
    
    for t in np.arange(0, T - delta_t, delta_t):
        y_data[:, k - 1] = swing(t, delta_t, y0_data, u[:, k - 1])
        y0_data = y_data[:, k - 1]
        k += 1
    
    U = np.append(U, np.fliplr(u).reshape((m * int(round(T / delta_t)), 1)), axis=1)
    X0 = np.append(X0, y0.reshape((n, 1)), axis=1)
    X1T = np.append(X1T, np.fliplr(y_data[:, :int(round(T / delta_t - 1))]).reshape((n * int(round(T / delta_t - 1)), 1)), axis=1)
    XT = np.append(XT, y_data[:, -1].reshape((n, 1)), axis=1)

# Compute data-driven control

U = U[:, :3750]
XT = XT[:, :3750]
X0 = X0[:, :3750]
X1T = X1T[:, :3750]

K_X0 = np.linalg.matrix_rank(X0, tol=1e-10)
K_U = np.linalg.matrix_rank(U, tol=1e-10)
xf_c = xf - np.dot(XT, np.dot(K_U, np.linalg.pinv(np.dot(X0, K_U), rcond=1e-10))) * x0

U = np.dot(U, K_X0)
X1T = np.dot(X1T, K_X0)
XT = np.dot(XT, K_X0)

K_U = np.linalg.matrix_rank(U, tol=1e-10)
K_XT = np.linalg.matrix_rank(XT, tol=1e-10)
Q = 5e-3
R = 1

L2 = Q * np.dot(X1T.T, X1T) + R * np.dot(U.T, U)
L = cholesky(L2, lower=True)
W, S, V = np.linalg.svd(np.dot(L, K_XT), full_matrices=False)
u_opt = np.dot(U, np.dot(np.linalg.pinv(XT, rcond=1e-10), xf_c)) - np.dot(U, np.dot(K_XT, np.dot(np.linalg.pinv(np.dot(W, np.dot(S, V.T)), rcond=1e-10), np.dot(L, np.dot(np.linalg.pinv(XT, rcond=1e-10), xf_c)))))
u_opt_seq = np.fliplr(u_opt.reshape((m, int(round(T / delta_t)))))
import numpy as np
import matplotlib.pyplot as plt

# Apply control for fault recovery

u_opt_seq = np.concatenate((np.zeros((9, int(round(t_c / delta_t) + 1))), u_opt_seq, np.zeros((9, int(round(500 / delta_t))))), axis=1)
y0 = np.zeros(18)

k = 1
tspan = np.arange(0, 500 + delta_t, delta_t)
yy = np.zeros((18, len(tspan)))

for t in tspan:
    yy[:, k - 1] = swing(t, delta_t, y0, u_opt_seq[:, k - 1])
    y0 = yy[:, k - 1]
    k += 1

# Plot results
plt.figure(1)

plt.subplot(2, 2, 2)
plt.plot(np.arange(0, T_sim + delta_t, delta_t), yy[0:9, 0:int(round(T_sim / delta_t) + 1)])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-10, 15, 15, -10], 'r', alpha=0.25)
plt.fill_between([t_c, t_c, t_c + T, t_c + T], [-10, 15, 15, -10], 'g', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.subplot(2, 2, 4)
plt.plot(np.arange(0, T_sim + delta_t, delta_t), yy[9:18, 0:int(round(T_sim / delta_t) + 1)])
plt.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [-10, 15, 15, -10], 'r', alpha=0.25)
plt.fill_between([t_c, t_c, t_c + T, t_c + T], [-10, 15, 15, -10], 'g', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

# Asymptotic behavior
plt.figure(2)

plt.subplot(2, 1, 1)
plt.plot(tspan, yy[0:9, :])
plt.fill_between([2, 2, 2.4, 2.4], [-10, 15, 15, -10], 'r', alpha=0.25)
plt.fill_between([t_c, t_c, t_c + T, t_c + T], [-10, 15, 15, -10], 'g', alpha=0.25)
plt.ylabel('phase')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.subplot(2, 1, 2)
plt.plot(tspan, yy[9:18, :])
plt.fill_between([2, 2, 2.4, 2.4], [-10, 15, 15, -10], 'r', alpha=0.25)
plt.fill_between([t_c, t_c, t_c + T, t_c + T], [-10, 15, 15, -10], 'g', alpha=0.25)
plt.ylabel('frequency')
plt.xlabel('time t')
plt.legend(['gen 10', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7', 'gen 8', 'gen 9'])

plt.show()

# Define the swing function
def swing(t, delta_t, y, u):
    global H, P, D, E, Y3, Y32, T_fault_in, T_fault_end
    yp = np.zeros(18)
    # Implement your swing function here
    return yp
import numpy as np

def swing(t, delta_t, y, u):
    global H, P, D, E, Y3, Y32, T_fault_in, T_fault_end
    
    yp = np.zeros(18)
    
    k = np.array([
        delta_t * (1 / H[1, 1]) * (P[1] - np.real(Y[1, 1]) * E[1]**2 - E[1] * E[0] * np.real(Y[1, 0]) - E[1] * E[2] * np.real(Y[1, 2]) - E[1] * E[3] * np.real(Y[1, 3]) - E[1] * E[4] * np.real(Y[1, 4]) - E[1] * E[5] * np.real(Y[1, 5]) - E[1] * E[6] * np.real(Y[1, 6]) - E[1] * E[7] * np.real(Y[1, 7]) - E[1] * E[8] * np.real(Y[1, 8]) - E[1] * E[9] * np.real(Y[1, 9]) - E[1] * E[10] * np.real(Y[1, 10])),
        delta_t * (1 / H[2, 2]) * (P[2] - np.real(Y[2, 2]) * E[2]**2 - E[2] * E[0] * np.real(Y[2, 0]) - E[2] * E[1] * np.real(Y[2, 1]) - E[2] * E[3] * np.real(Y[2, 3]) - E[2] * E[4] * np.real(Y[2, 4]) - E[2] * E[5] * np.real(Y[2, 5]) - E[2] * E[6] * np.real(Y[2, 6]) - E[2] * E[7] * np.real(Y[2, 7]) - E[2] * E[8] * np.real(Y[2, 8]) - E[2] * E[9] * np.real(Y[2, 9]) - E[2] * E[10] * np.real(Y[2, 10])),
        delta_t * (1 / H[3, 3]) * (P[3] - np.real(Y[3, 3]) * E[3]**2 - E[3] * E[0] * np.real(Y[3, 0]) - E[3] * E[1] * np.real(Y[3, 1]) - E[3] * E[2] * np.real(Y[3, 2]) - E[3] * E[4] * np.real(Y[3, 4]) - E[3] * E[5] * np.real(Y[3, 5]) - E[3] * E[6] * np.real(Y[3, 6]) - E[3] * E[7] * np.real(Y[3, 7]) - E[3] * E[8] * np.real(Y[3, 8]) - E[3] * E[9] * np.real(Y[3, 9]) - E[3] * E[10] * np.real(Y[3, 10])),
        delta_t * (1 / H[4, 4]) * (P[4] - np.real(Y[4, 4]) * E[4]**2 - E[4] * E[0] * np.real(Y[4, 0]) - E[4] * E[1] * np.real(Y[4, 1]) - E[4] * E[2] * np.real(Y[4, 2]) - E[4] * E[3] * np.real(Y[4, 3]) - E[4] * E[5] * np.real(Y[4, 5]) - E[4] * E[6] * np.real(Y[4, 6]) - E[4] * E[7] * np.real(Y[4, 7]) - E[4] * E[8] * np.real(Y[4, 8]) - E[4] * E[9] * np.real(Y[4, 9]) - E[4] * E[10] * np.real(Y[4, 10])),
        delta_t * (1 / H[5, 5]) * (P[5] - np.real(Y[5, 5]) * E[5]**2 - E[5] * E[0] * np.real(Y[5, 0]) - E[5] * E[1] * np.real(Y[5, 1]) - E[5] * E[2] * np.real(Y[5, 2]) - E[5] * E[3] * np.real(Y[5, 3]) - E[5] * E[4] * np.real(Y[5, 4]) - E[5] * E[6] * np.real(Y[5, 6]) - E[5] * E[7] * np.real(Y[5, 7]) - E[5] * E[8] * np.real(Y[5, 8]) - E[5] * E[9] * np.real(Y[5, 9]) - E[5] * E[10] * np.real(Y[5, 10])),
        delta_t * (1 / H[6, 6]) * (P[6] - np.real(Y[6, 6]) * E[6]**2 - E[6] * E[0] * np.real(Y[6, 0]) - E[6] * E[1] * np.real(Y[6, 1]) - E[6] * E[2] * np.real(Y[6, 2]) - E[6] * E[3] * np.real(Y[6, 3]) - E[6] * E[4] * np.real(Y[6, 4]) - E[6] * E[5] * np.real(Y[6, 5]) - E[6] * E[7] * np.real(Y[6, 7]) - E[6] * E[8] * np.real(Y[6, 8]) - E[6] * E[9] * np.real(Y[6, 9]) - E[6] * E[10] * np.real(Y[6, 10])),
        delta_t * (1 / H[7, 7]) * (P[7] - np.real(Y[7, 7]) * E[7]**2 - E[7] * E[0] * np.real(Y[7, 0]) - E[7] * E[1] * np.real(Y[7, 1]) - E[7] * E[2] * np.real(Y[7, 2]) - E[7] * E[3] * np.real(Y[7, 3]) - E[7] * E[4] * np.real(Y[7, 4]) - E[7] * E[5] * np.real(Y[7, 5]) - E[7] * E[6] * np.real(Y[7, 6]) - E[7] * E[8] * np.real(Y[7, 8]) - E[7] * E[9] * np.real(Y[7, 9]) - E[7] * E[10] * np.real(Y[7, 10])),
        delta_t * (1 / H[8, 8]) * (P[8] - np.real(Y[8, 8]) * E[8]**2 - E[8] * E[0] * np.real(Y[8, 0]) - E[8] * E[1] * np.real(Y[8, 1]) - E[8] * E[2] * np.real(Y[8, 2]) - E[8] * E[3] * np.real(Y[8, 3]) - E[8] * E[4] * np.real(Y[8, 4]) - E[8] * E[5] * np.real(Y[8, 5]) - E[8] * E[6] * np.real(Y[8, 6]) - E[8] * E[7] * np.real(Y[8, 7]) - E[8] * E[9] * np.real(Y[8, 9]) - E[8] * E[10] * np.real(Y[8, 10])),
        delta_t * (1 / H[9, 9]) * (P[9] - np.real(Y[9, 9]) * E[9]**2 - E[9] * E[0] * np.real(Y[9, 0]) - E[9] * E[1] * np.real(Y[9, 1]) - E[9] * E[2] * np.real(Y[9, 2]) - E[9] * E[3] * np.real(Y[9, 3]) - E[9] * E[4] * np.real(Y[9, 4]) - E[9] * E[5] * np.real(Y[9, 5]) - E[9] * E[6] * np.real(Y[9, 6]) - E[9] * E[7] * np.real(Y[9, 7]) - E[9] * E[8] * np.real(Y[9, 8]) - E[9] * E[10] * np.real(Y[9, 10])),
        delta_t * (1 / H[10, 10]) * (P[10] - np.real(Y[10, 10]) * E[10]**2 - E[10] * E[0] * np.real(Y[10, 0]) - E[10] * E[1] * np.real(Y[10, 1]) - E[10] * E[2] * np.real(Y[10, 2]) - E[10] * E[3] * np.real(Y[10, 3]) - E[10] * E[4] * np.real(Y[10, 4]) - E[10] * E[5] * np.real(Y[10, 5]) - E[10] * E[6] * np.real(Y[10, 6]) - E[10] * E[7] * np.real(Y[10, 7]) - E[10] * E[8] * np.real(Y[10, 8]) - E[10] * E[9] * np.real(Y[10, 9]))
    ])
    
    for i in range(1, 11):
        yp[i] = y[i] + k[i]
    
    yp[11] = u
    
    if T_fault_in <= t <= T_fault_end:
        yp[11] = 1.0
    
    return yp

# Initialize system variables
H = np.array([0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
D = 2.0
E = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Y = np.array([
    [5.0 - 1j * 20.0, -1j * 15.0, -1j * 10.0, -1j * 5.0, -1j * 2.5, 0, 0, 0, 0, 0, 0],
    [-1j * 15.0, 5.0 - 1j * 15.0, -1j * 15.0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1j * 10.0, -1j * 15.0, 5.0 - 1j * 15.0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1j * 5.0, 0, 0, 5.0 - 1j * 10.0, -1j * 10.0, 0, 0, 0, 0, 0, 0],
    [-1j * 2.5, 0, 0, -1j * 10.0, 5.0 - 1j * 10.0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5.0 - 1j * 10.0, -1j * 10.0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1j * 10.0, 5.0 - 1j * 10.0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5.0 - 1j * 5.0, -1j * 5.0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1j * 5.0, 5.0 - 1j * 5.0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0 - 1j * 5.0, -1j * 5.0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1j * 5.0, 5.0 - 1j * 5.0]
], dtype=complex)

# Parameters
t_end = 10.0  # Simulation end time
delta_t = 0.01  # Time step
u = 1.0  # Reference voltage magnitude
T_fault_in = 2.0  # Fault-in time
T_fault_end = 4.0  # Fault-clear time

# Initialize arrays to store results
time_points = int(t_end / delta_t) + 1
results = np.empty((time_points, 12), dtype=complex)

# Initial conditions
results[0, 0] = 1.0  # Time
results[0, 1:11] = E  # Voltage magnitudes
results[0, 11] = u  # Reference voltage magnitude

# Time integration loop
for i in range(1, time_points):
    t = delta_t * i  # Current time
    
    # Perform RK4 integration step
    k1 = delta_t * np.array([
        0.0,  # Derivative for time
        0.0,  # Derivative for E1
        0.0,  # Derivative for E2
        0.0,  # Derivative for E3
        0.0,  # Derivative for E4
        0.0,  # Derivative for E5
        0.0,  # Derivative for E6
        0.0,  # Derivative for E7
        0.0,  # Derivative for E8
        0.0,  # Derivative for E9
        0.0  # Derivative for E10
    ])
    
    k2 = delta_t * np.array([
        1 / 2,
        0.0,  # Derivative for E1
        0.0,  # Derivative for E2
        0.0,  # Derivative for E3
        0.0,  # Derivative for E4
        0.0,  # Derivative for E5
        0.0,  # Derivative for E6
        0.0,  # Derivative for E7
        0.0,  # Derivative for E8
        0.0,  # Derivative for E9
        0.0  # Derivative for E10
    ])
    
    k3 = delta_t * np.array([
        1 / 2,
        0.0,  # Derivative for E1
        0.0,  # Derivative for E2
        0.0,  # Derivative for E3
        0.0,  # Derivative for E4
        0.0,  # Derivative for E5
        0.0,  # Derivative for E6
        0.0,  # Derivative for E7
        0.0,  # Derivative for E8
        0.0,  # Derivative for E9
        0.0  # Derivative for E10
    ])
    
    k4 = delta_t * np.array([
        1,
        0.0,  # Derivative for E1
        0.0,  # Derivative for E2
        0.0,  # Derivative for E3
        0.0,  # Derivative for E4
        0.0,  # Derivative for E5
        0.0,  # Derivative for E6
        0.0,  # Derivative for E7
        0.0,  # Derivative for E8
        0.0,  # Derivative for E9
        0.0  # Derivative for E10
    ])
    
    y = results[i - 1]
    
    # Update state using RK4 integration step
    k1 = rk4_derivatives(t, y, k1)
    k2 = rk4_derivatives(t + delta_t / 2, y + k1 / 2, k2)
    k3 = rk4_derivatives(t + delta_t / 2, y + k2 / 2, k3)
    k4 = rk4_derivatives(t + delta_t, y + k3, k4)
    
    results[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # Update state

# Extract time and voltage magnitudes from results
time = results[:, 0]
voltages = np.abs(results[:, 1:11])

# Plot the voltage magnitudes
plt.figure(figsize=(10, 6))
plt.plot(time, voltages)
plt.title('Voltage Magnitudes Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Voltage Magnitude')
plt.legend(['Bus 1', 'Bus 2', 'Bus 3', 'Bus 4', 'Bus 5', 'Bus 6', 'Bus 7', 'Bus 8', 'Bus 9', 'Bus 10'])
plt.grid(True)
plt.show()

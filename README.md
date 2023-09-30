Readme file:
NE_test_parameters.py: This file contains grid parameters and is imported to set up the simulation.

Initialization:

y0: The initial state of the system, representing phase and frequency values for each generator.
delta_t: The sampling time for the simulation.
T_sim: The final simulation time.
tspan: Vector of simulation times.
T_fault_in and T_fault_end: Time range for introducing a fault.
m and n: Number of inputs and states in the system.
T: Control horizon.
xf: The final stable state.
t_c: Control starting time.
Swing Equation Simulation:

swing(t, delta_t, y, u): Function to simulate the swing equations.
Loop to simulate the fault and update the system state.
Control Strategy:

Calculation of control inputs to recover the system from a fault.
Plotting Results:

Various plots to visualize the system behavior.
follow these steps to use this code for your power grid simulation:

you should have installed the required dependencies mentioned in the Dependencies section.

Configure the grid parameters in the NE_test_parameters.py file.

Run the code.
The code provides various plots to visualize the system's behavior, including phase and frequency plots over time. These plots help you understand the system's response to faults and control inputs.

For further customization or improvements, you can replace the simple swing equation implementation (swing(t, delta_t, y, u)) with your own power grid model.

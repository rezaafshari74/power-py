Readme file
This Python script is designed for analyzing power systems under fault conditions. It calculates and visualizes bus currents after a fault has occurred in a power network.
Features
•	Calculates bus currents after a fault in a power network.
•	Visualizes the real and imaginary parts of the calculated currents.
•	Easy-to-use Python script for power system engineers and students.
Prerequisites
•	Before you can use this script, ensure you have the following:
•	Python 3.x installed on your machine.
•	Required Python packages (NumPy, Matplotlib) installed. You can install them using pip:
pip install numpy matplotlib
To get started with this project, follow these steps:
git clone https://github.com/your-username/power-fault-analysis.git
cd power-fault-analysis
Open the power_fault_analysis.py script in a text editor or Python IDE to customize the power system parameters. ( I ran the code in this 
Visual Studio Code program).
In the script, you can define the power system parameters, including:

Voltage values at bus 3 and bus 4 (V3 and V4 arrays).
Admittance matrices Y3 and Y4.
Fault impedance (Zf).
Current injection at bus 3 (I3 array).
Modify the parameters according to your specific power system scenario.

Run the script to calculate the bus currents:
python power_fault_analysis.py
Results
After running the script, it will print the calculated bus currents to the console. Additionally, it will create a bar chart to visualize the real and imaginary parts of the currents.

Contributing
Contributions to this project are welcome! If you have improvements or bug fixes to propose, please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix.

Make your changes and commit them.

Push your branch to your fork.

Create a pull request to merge your changes into this repository.

plt.xlabel('Bus Number')
plt.ylabel('Current (A)')
plt.title('Bus Currents After Fault')
plt.legend()
plt.show()


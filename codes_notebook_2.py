# SEQUOIA DEMONSTRATOR 
# Energy Use Case: Optimization of Charging Schedules for Electric Cars 
# Author: Andreas Sturm, andreas.sturm@iao.fraunhofer.de
# Date: 2022-12-09

from qiskit.circuit.library.n_local import QAOAAnsatz
from codes_notebook_1 import generate_example as generate_example_notebook_1


def generate_example():
    charging_unit, car_green, qcio, converter, qubo, number_binary_variables, qubo_minimization_result = generate_example_notebook_1()
    
    ising, ising_offset = qubo.to_ising()
    
    qaoa_reps = 2
    qaoa_circuit = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps)
    qaoa_circuit.measure_all()

    return charging_unit, car_green, qcio, converter, qubo, number_binary_variables, qubo_minimization_result, ising, ising_offset, qaoa_reps, qaoa_circuit
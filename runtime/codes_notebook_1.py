# SEQUOIA DEMONSTRATOR 
# Energy Use Case: Optimization of Charging Schedules for Electric Cars 
# Author: Andreas Sturm, andreas.sturm@iao.fraunhofer.de
# Date: 2022-12-09

from typing import List, Union
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramConverter, LinearEqualityToPenalty, IntegerToBinary
from qiskit_optimization.algorithms import CplexOptimizer

## --- Codes from Notebook 1 --- ##

class Car:
    def __init__(
        self,
        car_id: str, # an arbitrary name for the car
        time_slots_at_charging_unit: List[int], # time slots when the car is at the charging unit
        required_energy: int # energy units that should be charged
    ) -> None:
        self.car_id = car_id
        self.time_slots_at_charging_unit = time_slots_at_charging_unit
        self.required_energy = required_energy

    def __str__(self) -> str:
        return f"Car '{self.car_id}':\n" \
            f"  at charging station at time slots {self.time_slots_at_charging_unit}\n" \
            f"  requires {self.required_energy} energy units"
    
class ChargingUnit:
    def __init__(
        self,
        charging_unit_id: str, # an arbitrary name for the charging unit
        number_charging_levels: int,
        number_time_slots: int,
    ) -> None:
        self.charging_unit_id = charging_unit_id
        self.number_charging_levels = number_charging_levels
        self.number_time_slots = number_time_slots
        self.cars_to_charge = []

    def __str__(self) -> str:
        info_cars_registered = ""
        for car in self.cars_to_charge:
            info_cars_registered = info_cars_registered + " " + car.car_id
        return "Charging unit with\n" \
            "  charging levels: " + str(list(range(self.number_charging_levels)))[1:-1] + "\n" \
            "  time slots: " + str(list(range(self.number_time_slots)))[1:-1] + "\n" \
            "  cars to charge:" + info_cars_registered

    def register_car_for_charging(self, car: Car) -> None:
        if max(car.time_slots_at_charging_unit) > self.number_time_slots - 1:
            raise ValueError("From car required time slots not compatible with charging unit.")
        self.cars_to_charge.append(car)

    def reset_cars_for_charging(self) -> None:
        self.cars_to_charge = []

    def generate_constraint_matrix(self) -> np.ndarray:
        """Matrix with ones for times when car is at charging station
         and with zeros if car is not at charging station"""
        number_cars_to_charge = len(self.cars_to_charge)
        constraint_matrix = np.zeros(
            (number_cars_to_charge, number_cars_to_charge*self.number_time_slots))
        for row_index in range(0, number_cars_to_charge):
            offset = row_index*self.number_time_slots
            cols = np.array(self.cars_to_charge[row_index].time_slots_at_charging_unit)
            constraint_matrix[row_index, offset+cols] = 1
        return constraint_matrix

    def generate_constraint_rhs(self) -> np.ndarray:
        """Vector with required energy as entries"""
        number_cars_to_charge = len(self.cars_to_charge)
        constraint_rhs = np.zeros((number_cars_to_charge, 1))
        for row_index in range(0, number_cars_to_charge):
            constraint_rhs[row_index] = self.cars_to_charge[row_index].required_energy
        return constraint_rhs

    def generate_cost_matrix(self) -> np.ndarray:
        number_cars_to_charge = len(self.cars_to_charge)
        return np.kron(
            np.ones((number_cars_to_charge, 1)) @ np.ones((1, number_cars_to_charge)),
            np.eye(self.number_time_slots))
    
def generate_qcio(
    charging_unit: ChargingUnit,
    name: str=None
) -> QuadraticProgram:
    if name is None:
        name = ""
    qcio = QuadraticProgram(name)

    for car in charging_unit.cars_to_charge:
        qcio.integer_var_list(
            keys=[f"{car.car_id}_t{t}" for t in range(0, charging_unit.number_time_slots)],
            lowerbound=0,
            upperbound=charging_unit.number_charging_levels-1,
            name="power.")

    constraint_matrix = charging_unit.generate_constraint_matrix()
    constraint_rhs = charging_unit.generate_constraint_rhs()
    for row_index in range(0, constraint_matrix.shape[0]):
        qcio.linear_constraint(
            linear=constraint_matrix[row_index, :],
            rhs=constraint_rhs[row_index][0],
            sense="==",
            name=f"charge_correct_energy_for_{charging_unit.cars_to_charge[row_index].car_id}")

    cost_matrix = charging_unit.generate_cost_matrix()
    qcio.minimize(quadratic=cost_matrix)

    return qcio

class Converter(QuadraticProgramConverter):
    def __init__(
        self, 
        penalty: float=None # the penalty paramter for step 1
    ) -> None:
        super().__init__()
        self._penalty = penalty
        self.linear_equality_to_penalty_converter = LinearEqualityToPenalty(penalty)
        self.integer_to_binary_converter = IntegerToBinary()

    def convert(self, quadratic_program: QuadraticProgram) -> QuadraticProgram:
        return self.integer_to_binary_converter.convert(
            self.linear_equality_to_penalty_converter.convert(quadratic_program))
    
    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        return self.linear_equality_to_penalty_converter.interpret(
            self.integer_to_binary_converter.interpret(x))

## --- Example for Notebook 2 --- ##

def generate_example():
    charging_unit = ChargingUnit(
        charging_unit_id="charging_unit",
        number_charging_levels=4,
        number_time_slots=4)
    car_green = Car(
        car_id="car_green",
        time_slots_at_charging_unit=[0, 1, 2],
        required_energy=4)
    charging_unit.register_car_for_charging(car_green)

    qcio = generate_qcio(charging_unit, name="QCIO")
    converter = Converter(penalty=3.6)
    qubo = converter.convert(qcio)
    number_binary_variables = qubo.get_num_binary_vars()

    cplex_optimizer = CplexOptimizer()
    qubo_minimization_result = cplex_optimizer.solve(qubo)

    return charging_unit, car_green, qcio, converter, qubo, number_binary_variables, qubo_minimization_result
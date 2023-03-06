# SEQUOIA DEMONSTRATOR 
# Energy Use Case: Optimization of Charging Schedules for Electric Cars 
# Author: Andreas Sturm, andreas.sturm@iao.fraunhofer.de
# Date: 2022-12-09

from typing import Union, List
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import plotly.graph_objects as go
from qiskit.circuit import QuantumCircuit
from qiskit_optimization.algorithms.optimization_algorithm import OptimizationResult
import matplotlib.pyplot as plt


def plot_charging_schedule_mp(
        charging_unit, #: ChargingUnit
        minimization_result_x, # OptimizationResult.x,
        marker_size=50,
    ):
    marker_colors = ["green", "orange", "blue", "red", "magenta", "goldenrod"]
    time_slots = np.arange(0, charging_unit.number_time_slots)
    fig, ax = plt.subplots()
    already_in_legend = []
    for t in time_slots:
        offset = 0
        for car_num in np.arange(0, len(charging_unit.cars_to_charge)):
            car_id_current_car = charging_unit.cars_to_charge[car_num].car_id
            minimization_result_x_current_car = minimization_result_x[
                car_num*charging_unit.number_time_slots:(car_num+1)*charging_unit.number_time_slots]
            power_t = minimization_result_x_current_car[t]
            if power_t > 0:
                ax.scatter(
                    x=[t+0.5]*int(power_t),
                    y=offset + np.arange(0, power_t),
                    s=marker_size,
                    marker="s",
                    color=marker_colors[car_num],
                    label=car_id_current_car if car_id_current_car not in already_in_legend else None,
                )
                offset += power_t
                already_in_legend.append(car_id_current_car)

    ax.set_xlim(0, charging_unit.number_time_slots)
    ax.set_xticks(np.arange(0.5, charging_unit.number_time_slots))
    ax.set_xticklabels(np.arange(0, charging_unit.number_time_slots))
    ax.set_xlabel("time slot", fontsize=12)
    ax.set_ylim(-0.6, charging_unit.number_charging_levels-1)
    ax.set_yticks(np.arange(-0.5, charging_unit.number_charging_levels-0.5))
    ax.set_yticklabels(np.arange(0, charging_unit.number_charging_levels))
    ax.set_ylabel("charging level", fontsize=12)
    ax.grid(False)
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig







def plot_charging_schedule(
        charging_unit, #: ChargingUnit
        minimization_result_x, # OptimizationResult.x,
        marker_size=50,
    ) -> go.Figure:
    marker_colors = ["green", "orange", "blue", "red", "magenta", "goldenrod"]
    time_slots = np.arange(0, charging_unit.number_time_slots)
    fig = go.Figure()
    already_in_legend = []
    for t in time_slots:
        offset = 0
        for car_num in np.arange(0, len(charging_unit.cars_to_charge)):
            car_id_current_car = charging_unit.cars_to_charge[car_num].car_id
            minimization_result_x_current_car = minimization_result_x[
                car_num*charging_unit.number_time_slots:(car_num+1)*charging_unit.number_time_slots]
            power_t = minimization_result_x_current_car[t]
            fig.add_trace(go.Scatter(
                x=[t+0.5]*int(power_t),
                y=offset + np.arange(0, power_t),
                mode="markers",
                marker_symbol="square",
                marker_size=marker_size,
                marker_color=marker_colors[car_num],
                name=car_id_current_car,
                showlegend=False if car_id_current_car in already_in_legend else True
            ))
            offset += power_t
            if power_t > 0:
                already_in_legend.append(car_id_current_car)
    
    fig.update_xaxes(
        tick0=1,
        dtick=1,
        range=[0.01, charging_unit.number_time_slots],
        tickvals=np.arange(0.5, charging_unit.number_time_slots),
        ticktext=np.arange(0, charging_unit.number_time_slots),
        title="time slot",
        title_font_size=12,
    )
    fig.update_yaxes(
        range=[-0.6, charging_unit.number_charging_levels-1],
        tickvals=np.arange(-0.5, charging_unit.number_charging_levels-0.5),
        ticktext=np.arange(0, charging_unit.number_charging_levels),
        title="charging level",
        title_font_size=12,
        zeroline=False
    )
    return fig

def convert_to_date_and_time_string(time_stamp: Union[datetime, str]):
    if isinstance(time_stamp, datetime):
        output = str(time_stamp.year) + "_" + \
            str(time_stamp.month).rjust(2, '0') + "_" + \
            str(time_stamp.day).rjust(2, '0') + "-" + \
            str(time_stamp.hour).rjust(2, '0') + "h" + \
            str(time_stamp.minute).rjust(2, '0') + "m"
    elif isinstance(time_stamp, str):
        output = time_stamp[0:17].replace('-', '_').replace('T', '-').replace(':', 'h', 1).replace(':', 'm', 1)
    else:
        raise ValueError("data type of 'time_stamp' not supported")
    return output

def save_token(token: str, file_name: str):
    path_token_file = Path(file_name).with_suffix(".pickle")
    if path_token_file.exists():
        print("Token already saved.")
    else:
        with open(path_token_file, 'wb') as file:
            pickle.dump(token, file)
        print(f"Token has been saved in '{file_name}.pickle'.")
        
def load_token(file_name: str):
    path_token_file = Path(file_name).with_suffix(".pickle")
    try:
        with open(path_token_file, 'rb') as file:
            token = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Token has not been saved. Use the function save_token to to save your token.")
    print("Token loaded.")
    return token

def count_gates(
        quantum_circuit: QuantumCircuit,
        gates_to_consider: List[str]
    ) -> int:
    result = 0
    for gate in gates_to_consider:
        try:
            count_gate = quantum_circuit.count_ops()[gate]
        except KeyError:
            count_gate = 0
        result = result + count_gate
    return result
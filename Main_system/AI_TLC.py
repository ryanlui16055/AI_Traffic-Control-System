import os
from tkinter import filedialog, ttk
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from ultralytics import YOLO
from screeninfo import get_monitors
import tkinter as tk
import torch
import statistics
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# -------------------sumo library part------------------#
import sumolib
import traci
from datetime import datetime, timedelta
import traci._simulation
import traci._vehicletype

# -------------------INIT parameter-------------------#

add_ons_time = 0
Scale_traffic_factor = "1"

yolo_path = os.path.join(os.getcwd(), "Main_system", "best.pt")
yolo_model = YOLO(yolo_path)

GRU_path = os.path.join(
    os.getcwd(), "LSTM_model", "Model", "best_tuned_GRU_model_30_time_step.keras"
)
GRU_model = tf.keras.models.load_model(GRU_path)
# -------------------Support Function-------------------#

# return height and weight
def get_monitor_info():
    return get_monitors()[0].height, get_monitors()[0].width

def get_YOLO_result(image):

    torch.cuda.set_device(0)
    
    results = yolo_model(image)[0]
    high_conf_boxes = [box for box in results.boxes if box.conf >= 0.4]

    # If there are boxes with high confidence, display and save the result
    if high_conf_boxes and high_conf_boxes!=[]:
        result_with_high_conf = results  # Create a copy of the result
        result_with_high_conf.boxes = (
            high_conf_boxes  # Assign only high confidence boxes
        )
        an_img = result_with_high_conf.plot()
        num_detected = len(high_conf_boxes)
    else:
        num_detected = 0
        an_img = image 

    return an_img, num_detected  # return image , number of detected car

def get_GRU_result(data):
    re = GRU_model.predict(data)
    return re[-1][0]

def get_current_simTime(simulation_start_time):
    # Get the current simulation time (in milliseconds)
    current_simulation_time_s = traci.simulation.getTime()

    # Calculate the current time (HH:MM) in the simulation
    current_time_in_simulation = simulation_start_time + timedelta(
        seconds=current_simulation_time_s + add_ons_time
    )

    # Format the current simulation time as HH
    current_time_formatted = current_time_in_simulation.strftime("%H")

    return int(current_time_formatted)

def set_add_ons_time(time):
    global add_ons_time
    add_ons_time = add_ons_time + int(time) * 3600

def set_scale_traffic(num):
    global Scale_traffic_factor
    Scale_traffic_factor = num


def nor_data(data):
    data_df = pd.DataFrame(data)

    data_df["hour_sin"] = np.sin(data_df["hour"] * 2 * np.pi / 24)
    data_df["hour_cos"] = np.cos(data_df["hour"] * 2 * np.pi / 24)
    data_df.drop(["hour"],axis=1,inplace=True,)

    data = data_df["sampleSize"]
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, 75)

    # Define a function to categorize the data based on the quartiles
    def categorize(value):
        if value < Q1:
            return 1
        elif Q1 <= value <= Q2:
            return 2
        elif Q2 < value <= Q3:
            return 2
        else:
            return 3

    # Apply the categorization function to the 'sampleSize' column
    data_df["sampleSize"] = data_df["sampleSize"].apply(categorize)

    data_df = pd.get_dummies(data_df, columns=["sampleSize"], dtype="float")


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_test = pd.DataFrame(
        scaler.fit_transform(data_df), columns=data_df.columns
    )
    dummy_targets = np.zeros((scaled_data_test.shape[0], 1))
    generator_test = TimeseriesGenerator(
        scaled_data_test, dummy_targets, length=30, batch_size=1
    )
    return generator_test

# --------------------Class------------------------------#
class VehicleTracker:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.vehicles = {}
        self.vehicles_speed = []
        self.DateStore = []
    def is_in_area(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def update(self, step, time):
        for veh_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            in_area = self.is_in_area(x, y)

            if veh_id not in self.vehicles and in_area:
                # Vehicle has entered the area
                self.vehicles[veh_id] = {"enter_time": step, "exit_time": None}
                self.vehicles_speed.append(speed)
            elif (
                veh_id in self.vehicles
                and not in_area
                and self.vehicles[veh_id]["exit_time"] is None
            ):
                # Vehicle has exited the area
                self.vehicles[veh_id]["exit_time"] = step
        if (step > 1):
            self.store_data(time)

    def store_data(self,time):
        speed = self.get_Percent_Speed()
        self.DateStore.append(
            {
                "medianSpeed": self.get_median_Speed(),
                "averageSpeed": self.get_average_Speed(),
                "standardDeviationSpeed": self.get_STDEV_Speed(),
                "travelTimeStandardDeviation": self.get_STDEV_travel_time(),
                "sampleSize": self.get_vehicle_count(),
                "averageTravelTime": self.get_average_travel_time(),
                "medianTravelTime": self.get_median_travel_time(),
                "speedPercentiles_25": speed[0],
                "speedPercentiles_50": speed[1],
                "speedPercentiles_75": speed[2],
                "speedPercentiles_95": speed[3],
                "hour": time,
            }
        )

    def get_median_travel_time(self):
        re = []
        for veh_id, data in self.vehicles.items():
            if data["exit_time"] is not None:
                travel_time = data["exit_time"] - data["enter_time"]
                re.append(travel_time)

        if len(re) <2:
            re = [8.5,8.5]
        return statistics.median(re)

    def get_average_travel_time(self):
        re = []
        for veh_id, data in self.vehicles.items():
            if data["exit_time"] is not None:
                travel_time = data["exit_time"] - data["enter_time"]
                re.append(travel_time)

        if len(re) <2:
            re = [8.5,8.5]
        return statistics.mean(re)

    def get_STDEV_travel_time(self):
        re = []
        for veh_id, data in self.vehicles.items():
            if data["exit_time"] is not None:
                travel_time = data["exit_time"] - data["enter_time"]
                re.append(travel_time)
        if len(re) <2:
            re = [8.5,8.5]
        return statistics.stdev(re)

    def get_median_Speed(self):
        re = self.vehicles_speed
        if len(re) < 2:
            re = [8.5,8.5]
        return statistics.median(re)

    def get_average_Speed(self): 
        re = self.vehicles_speed
        if len(re) < 2:
            re = [20,20]
        return statistics.mean(re)

    def get_STDEV_Speed(self):
        re = self.vehicles_speed
        if len(re) < 2:
            re = [20,20]
        return statistics.stdev(re)

    def get_Percent_Speed(self):
        res = self.vehicles_speed
        if len(res) < 2:
            res = [20, 20]

        P_25 = np.percentile(res, 25)
        P_50 = np.percentile(res, 50)
        P_75 = np.percentile(res, 75)
        P_95 = np.percentile(res, 95)
        re = [P_25, P_50, P_75, P_95]
        return re

    def get_vehicle_count(self):
        return len(self.vehicles)

# ------------------------------------------Main Function-----------------------------------------------#

def SUMO_INIT_2(
    sumo_cfg_path,
    win_pos,
    win_size,
    delay,
    zoom_lv=1000,
    off_set=None,
    simulation_steps=1,
):
    screenshot_path = os.path.join(os.getcwd(), "LSTM_model", "image", "sumo_screenshot.png")
    monitor_h, monitor_w = get_monitor_info()

    win_name = "TLC"
    cv2.namedWindow(win_name)  # Create a named window
    cv2.moveWindow(win_name, int(monitor_w * 0.55), monitor_h // 5)

    sumo_gui_name = "sumo-gui"
    sumobin = sumolib.checkBinary(sumo_gui_name)

    # check if sumo-gui installed
    if sumobin == sumo_gui_name:
        raise SyntaxError

    sumoCmd = [
        sumobin,
        "-c",
        sumo_cfg_path,
        "--window-pos",
        win_pos,
        "--window-size",
        win_size,
        "--delay",
        delay,
        # "--step-length",
        # "5",
        "--scale",
        Scale_traffic_factor,
        "--start",
    ]

    traci.start(sumoCmd)

    # Set zoom level (scale)
    traci.gui.setZoom("View #0", zoom_lv)  # Zoom level 1000

    if off_set is not None:
        traci.gui.setOffset("View #0", off_set[0], off_set[1])

    # Set the simulation start time
    start_time = datetime(2024, 8, 15, 0, 0, 0)

    tracker = VehicleTracker(x_min=0, y_min=0, x_max=500, y_max=500)

    # Run the simulation steps and update the Tkinter window
    for _ in range(simulation_steps):

        traci.gui.screenshot("View #0", screenshot_path)
        traci.simulationStep()
        current_time = get_current_simTime(start_time)
        tracker.update(_, current_time)

        # cv2 version
        img = cv2.imread(screenshot_path)
        tk_YOLO_image, num_obj = get_YOLO_result(img)
        cv2.imshow(win_name, tk_YOLO_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # ------------TLC PART------------#

        print(traci.trafficlight.getRedYellowGreenState("GS_cluster_109414_9030029183"))
        print("-----------------------")
        if _ % 90 == 0 and _ != 0:
            current_time = get_current_simTime(start_time)
            GRU_result = get_GRU_result(nor_data(tracker.DateStore))
            SUMO_algorithm_TLC("GS_cluster_109414_9030029183", num_obj, GRU_result,40)
            print(tracker.DateStore[-1]["averageSpeed"])

    traci.close()


def SUMO_algorithm_TLC(tl_id, YOLO_value, GRU_value, max_vehicles):
    YOLO_weight = 0.6
    RNN_weight = 0.4
    # Normalize YOLO output (number of vehicles) to a scale of 0 to 1
    normalized_yolo = max(YOLO_value / max_vehicles, 0.1)

    if GRU_value > 0.75:
        YOLO_weight = 0.4
        RNN_weight = 0.6

    # Combine YOLO and LSTM outputs using the weights
    combined_output = (YOLO_weight * normalized_yolo) + (RNN_weight * GRU_value)

    # Adjust the green light duration based on the combined output
    # The duration is increased if the combined output suggests high congestion
    green_light_duration = 20 * (
        1 + combined_output
    )  # Increase duration based on combined output

    # Ensure green light duration stays within practical limits
    green_light_duration = np.clip(
        green_light_duration, 10, 120
    )  # Clip between 10 and 120 seconds

    # ------------------------------------------------

    def create_phase(duration, state, min_dur=None, max_dur=None):
        return traci.trafficlight.Phase(duration, state, min_dur, max_dur)

    # Define your custom phases
    phases = [
        create_phase(green_light_duration, "GGGrrrGGGrrr"),  # North-South Green
        create_phase(3, "yyyrrryyyrrr"),  # North-South Yellow
        create_phase(green_light_duration, "rrrGGGrrrGGG"),  # East-Wqest Green
        create_phase(3, "rrryyyrrryyy"),  # East-West Yellow
        create_phase(green_light_duration, "GGGrrrGGGrrr"),  # North-South Green
        create_phase(3, "yyyrrryyyrrr"),  # North-South Yellow
    ]

    logic = traci.trafficlight.Logic("0", 0, 0, phases)

    traci.trafficlight.setProgramLogic(tl_id, logic)

    # Create the program
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, logic)
    traci.trafficlight.setProgram(tl_id, logic.programID)

    traci.trafficlight.setPhase(tl_id, 0)


def tk_menu():

    root = tk.Tk()
    root.title("AI_TLC_SIMULATION")

    # Row 1: File browser to select a file
    def browse_file():
        file_path.set(filedialog.askopenfilename())

    file_path = tk.StringVar()
    tk.Label(root, text="File path of .cfg:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    tk.Entry(root, textvariable=file_path, width=25).grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=5)

    # Row 2: Entry for 'Delay'
    tk.Label(root, text="Delay(ms):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    delay_entry = tk.Entry(root, width=25)
    delay_entry.grid(row=1, column=1, padx=10, pady=5)

    # Row 3: Entry for 'Simulation Step'
    tk.Label(root, text="Simulation Step:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    simulation_step_entry = tk.Entry(root, width=25)
    simulation_step_entry.grid(row=2, column=1, padx=10, pady=5)

    # Row 4: Dropdown for selecting time in hour (00:00 to 23:00)
    tk.Label(root, text="Time:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    time_var = tk.StringVar(value="00:00")
    time_options = [f"{hour:02d}:00" for hour in range(24)]
    time_dropdown = ttk.Combobox(
        root, textvariable=time_var, values=time_options, state="readonly", width=23
    )
    time_dropdown.grid(row=3, column=1, padx=10, pady=5)

    # Row 5: Entry for 'Scale Traffic Factor'
    tk.Label(root, text="Scale Traffic Factor:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    scale_traffic_factor_entry = tk.Entry(root, width=25)
    scale_traffic_factor_entry.grid(row=4, column=1, padx=10, pady=5)

    # Row 6: Submit Button
    def Start_Simulation():
        print("File:", file_path.get())
        print("Delay:", delay_entry.get())
        print("Simulation Step:", simulation_step_entry.get())
        print("Time:", time_var.get())
        print("Scale Traffic Factor:", scale_traffic_factor_entry.get())

        monitor_h, monitor_w = get_monitor_info()
        # start SUMO simulation
        sumo_cfg = file_path.get()

        win_pos = "0,"+ str(monitor_h // 5)
        win_size = str(int(monitor_w * 0.45)) + "," + str(int(monitor_h * 0.65))

        delay = delay_entry.get()
        zoom_lv = "600"
        off_set = [320.14, 174.36]
        simulation_steps = int(simulation_step_entry.get())

        set_add_ons_time(time_var.get()[:2])
        set_scale_traffic(scale_traffic_factor_entry.get())

        root.destroy()

        SUMO_INIT_2(
            sumo_cfg,
            win_pos,
            win_size,
            delay,
            zoom_lv,
            off_set,
            simulation_steps,
        )

    submit_button = tk.Button(root, text="Start Simulation", command=Start_Simulation)
    submit_button.grid(row=5, column=1, pady=10)

    # Start the GUI loop
    root.mainloop()


def main():
    tk_menu()

if __name__ == "__main__":
    main()

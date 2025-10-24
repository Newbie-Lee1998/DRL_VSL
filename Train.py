import sys
import traci
import numpy as np
from sumolib import checkBinary
import os
import matplotlib.pyplot as plt
import torch
from SAC_Agent import SACAgent
import datetime
import random

class SumoEnv:
    def __init__(self, if__show__gui, sumo_cfg):
        self.sumo_cfg = sumo_cfg
        self.lane_count = 3
        self.control_interval = 120
        self.preheating_time = 600
        self.vehicle_stopped = False
        self.bottle_lane_ids = ['E8_0', 'E8_1', 'E8_2']
        self.acc_lane_ids = ['E7_0', 'E7_1', 'E7_2']
        self.core_lane_ids = ["E6_0", "E6_1", "E6_2"]
        self.out_detector_ids = ["bottle_down_0", "bottle_down_1"]
        self.in_detector_ids = ["bottle_up_0", "bottle_up_1", "bottle_up_2"]
        self.speed_limits = {}
        self.processed_vehicles = set()
        self.step = 0
        self.simulation_step = 0
        self.if__show__gui = if__show__gui
        if not if__show__gui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

    def get_traffic_in_3lanes(self):
        up_2zone_traffic = []
        num_vehicles_E1_0 = traci.lane.getLastStepVehicleNumber("E4_0")
        num_vehicles_E2_0 = traci.lane.getLastStepVehicleNumber("E5_0")
        num_vehicles_E1_1 = traci.lane.getLastStepVehicleNumber("E4_1")
        num_vehicles_E2_1 = traci.lane.getLastStepVehicleNumber("E5_1")
        num_vehicles_E1_2 = traci.lane.getLastStepVehicleNumber("E4_2")
        num_vehicles_E2_2 = traci.lane.getLastStepVehicleNumber("E5_2")
        sum_0 = num_vehicles_E1_0 + num_vehicles_E2_0
        sum_1 = num_vehicles_E1_1 + num_vehicles_E2_1
        sum_2 = num_vehicles_E1_2 + num_vehicles_E2_2
        up_2zone_traffic.append(sum_0)
        up_2zone_traffic.append(sum_1)
        up_2zone_traffic.append(sum_2)
        return up_2zone_traffic

    def get_occ_in_3lanes(self, lane_ids):
        occupancies = []
        for lane_id in lane_ids:
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            occupancies.append(occupancy * 100)
        return occupancies

    def get_average_speed_in_3lanes(self, lane_ids):
        lanes_speeds = []
        for lane_id in lane_ids:
            lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            lanes_speeds.append(lane_speed)
        return lanes_speeds

    def get_lane_limited_speeds(self):
        lane_limited_speeds = [traci.lane.getMaxSpeed(lane_id) for lane_id in self.core_lane_ids]
        return lane_limited_speeds

    def get_vehicle_counts(self, detector_ids):
        evacuated_counts = sum(
            [traci.inductionloop.getLastStepVehicleNumber(detector_id) for detector_id in detector_ids])
        return evacuated_counts

    def hdvsl(self, action, increment=2.78, max_speed=27.78):
        vsl_control_edges = ['E6', 'E5', 'E4', 'E3', 'E2', 'E1']
        current_limits = action.copy()
        # lane 0
        lane0_vsl_count = 0
        lane0_vsl = []
        lane0_vsl.append(current_limits[0])
        while action[0] < max_speed:
            action[0] += increment
            lane0_vsl.append(action[0])
            lane0_vsl_count += 1
        for i in range(lane0_vsl_count):
            edge = vsl_control_edges[i]
            traci.lane.setMaxSpeed(f'{edge}_{0}', lane0_vsl[i])
        # lane 1
        lane1_vsl_count = 0
        lane1_vsl = []
        lane1_vsl.append(current_limits[1])
        while action[1] < max_speed:
            action[1] += increment
            lane1_vsl.append(action[1])
            lane1_vsl_count += 1
        for i in range(lane1_vsl_count):
            edge = vsl_control_edges[i]
            traci.lane.setMaxSpeed(f'{edge}_{1}', lane1_vsl[i])
        # lane 2
        lane2_vsl_count = 0
        lane2_vsl = []
        lane2_vsl.append(current_limits[2])
        while action[2] < max_speed:
            action[2] += increment
            lane2_vsl.append(action[2])
            lane2_vsl_count += 1
        for i in range(lane2_vsl_count):
            edge = vsl_control_edges[i]
            traci.lane.setMaxSpeed(f'{edge}_{2}', lane2_vsl[i])

    def lane_change_guidance(self, veh_id, target_lane):
        traci.vehicle.changeLane(veh_id, target_lane, 3)

    def start_simulation(self):
        seed = 1010
        self.step = 0
        self.simulation_step = 0
        self.vehicle_stopped = False
        self.speed_limits = {}

        self.processed_vehicles = set()
        sumo_cmd = [self.sumoBinary, "-c", self.sumo_cfg, '--seed', str(seed)]
        traci.start(sumo_cmd)

    def close(self):
        traci.close()

    def simulation_preheating(self, v_list):
        self.speed_limits[self.step] = v_list
        total_entry_vehicles = 0
        total_evacuate_vehicles = 0

        q_2zone_lane = [0] * self.lane_count
        occ_acc_lane = [0] * self.lane_count
        speed_acc_lane = [0] * self.lane_count
        occ_bottle_list = [0] * self.lane_count
        sample_count = 0
        for step in range(self.preheating_time):
            traci.simulationStep()
            self.simulation_step += 1
            total_evacuate_vehicles += self.get_vehicle_counts(self.out_detector_ids)
            total_entry_vehicles += self.get_vehicle_counts(self.in_detector_ids)
            """20s"""
            if step % 20 == 0:
                q_2zone_lane = [x + y for x, y in zip(q_2zone_lane, self.get_traffic_in_3lanes())]
                occ_acc_lane = [x + y for x, y in zip(occ_acc_lane, self.get_occ_in_3lanes(self.acc_lane_ids))]
                speed_acc_lane = [x + y for x, y in zip(speed_acc_lane, self.get_average_speed_in_3lanes(self.acc_lane_ids))]
                occ_bottle_list = [x + y for x, y in zip(occ_bottle_list, self.get_occ_in_3lanes(self.bottle_lane_ids))]
                sample_count += 1
        dif = (total_evacuate_vehicles - total_entry_vehicles)
        q_2zone_lane = [x / sample_count for x in q_2zone_lane]
        q_2zone_lane = [round(x) for x in q_2zone_lane]
        occ_acc_lane = [x / sample_count for x in occ_acc_lane]
        speed_acc_lane = [x / sample_count for x in speed_acc_lane]
        occ_bottle_list = [x / sample_count for x in occ_bottle_list]
        state_interval = q_2zone_lane + occ_acc_lane + speed_acc_lane + occ_bottle_list
        print("*********************Preheating_end*****************************")
        return state_interval, dif, self.simulation_step

    def run_one_interval(self, v_list):
        self.step += 1
        self.speed_limits[self.step] = v_list
        total_entry_vehicles = 0
        total_evacuate_vehicles = 0
        q_2zone_lane = [0] * self.lane_count
        occ_acc_lane = [0] * self.lane_count
        speed_acc_lane = [0] * self.lane_count
        occ_bottle_list = [0] * self.lane_count
        sample_count = 0
        if self.vehicle_stopped:
            self.hdvsl(v_list)

        for step in range(1, self.control_interval + 1):
            traci.simulationStep()
            self.simulation_step += 1
            if 'accident_vehicle' in traci.vehicle.getIDList():
                if traci.vehicle.getRoadID('accident_vehicle') == "E5" and not self.vehicle_stopped:
                    traci.vehicle.setStop('accident_vehicle', edgeID='E8', pos=500, laneIndex=2)
                    print(
                        f"———>>>Vehicle 'accident_vehicle' stopped at position {'E8_500'} at step {self.simulation_step} (time {self.simulation_step} seconds).")
                    self.vehicle_stopped = True
            if self.simulation_step >= 660:
                vehicles_E8_2 = traci.lane.getLastStepVehicleIDs('E8_2')
                vehicles_E8_1 = traci.lane.getLastStepVehicleIDs('E8_1')
                for veh_id in vehicles_E8_2:
                    if veh_id not in self.processed_vehicles and random.random() < 0.8:
                        self.lane_change_guidance(veh_id, 1)
                        self.processed_vehicles.add(veh_id)
                for veh_id in vehicles_E8_1:
                    if veh_id not in self.processed_vehicles and random.random() < 0.8:
                        self.lane_change_guidance(veh_id, 0)
                        self.processed_vehicles.add(veh_id)
            total_evacuate_vehicles += self.get_vehicle_counts(self.out_detector_ids)
            total_entry_vehicles += self.get_vehicle_counts(self.in_detector_ids)
            if step % 20 == 0:
                q_2zone_lane = [x + y for x, y in zip(q_2zone_lane, self.get_traffic_in_3lanes())]
                occ_acc_lane = [x + y for x, y in zip(occ_acc_lane, self.get_occ_in_3lanes(self.acc_lane_ids))]
                speed_acc_lane = [x + y for x, y in zip(speed_acc_lane, self.get_average_speed_in_3lanes(self.acc_lane_ids))]
                occ_bottle_list = [x + y for x, y in zip(occ_bottle_list, self.get_occ_in_3lanes(self.bottle_lane_ids))]
                sample_count += 1
        dif = (total_evacuate_vehicles - total_entry_vehicles)
        q_2zone_lane = [x / sample_count for x in q_2zone_lane]
        q_2zone_lane = [round(x) for x in q_2zone_lane]
        occ_acc_lane = [x / sample_count for x in occ_acc_lane]
        speed_acc_lane = [x / sample_count for x in speed_acc_lane]
        occ_bottle_list = [x / sample_count for x in occ_bottle_list]
        state_interval = q_2zone_lane + occ_acc_lane + speed_acc_lane + occ_bottle_list
        print("*******************************ep_step：{} *******************************：".format(self.step), dif)
        return state_interval, dif, self.simulation_step


def train_sac_agent(sac_agent, Max_ep, total_simulation_time):
    all_ep_reward = []
    sumo_cfg_file = r"D:/SUMO_install/sumo-1.20.0/My_file/simulation/simulation.sumocfg"
    env = SumoEnv(False, sumo_cfg_file)
    for ep in range(Max_ep):
        env.start_simulation()
        v_list = [27.78, 27.78, 27.78]
        ep_reward = 0
        ep_step_count = 0
        state, reward, simulationSteps = env.simulation_preheating(v_list)
        while simulationSteps < total_simulation_time:
            action = sac_agent.select_action(state)
            action = [v * (5 / 18) for v in action]
            print("vsl_value:", action)
            next_state, reward, simulationSteps = env.run_one_interval(action)
            done = simulationSteps >= total_simulation_time or traci.vehicle.getIDCount() == 0
            sac_agent.buffer.add(state, action, reward, next_state, done)
            sac_agent.update()
            state = next_state
            ep_reward += reward
            ep_step_count += 1
            if done:
                break
        all_ep_reward.append(ep_reward)
        print(f"————Episode: {ep}, Ep_steps: {ep_step_count}, Episode Reward: {ep_reward}")
        env.close()
    print("=========================================== Over===========================================")
    print(all_ep_reward)
    plt.plot(np.arange(len(all_ep_reward)), all_ep_reward)
    plt.title('xxx')
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()


if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tool')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    Max_ep = 600
    total_simulation_time = 7200
    state_dim = 12
    action_dim = 3
    buffer_size = 10000
    batch_size = 128
    actor_lr = 1e-4
    critic_lr = 1e-4
    alpha_lr = 3e-4
    target_entropy = -3.0
    gamma = 0.9
    tau = 0.005
    bl = 40
    br = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(state_dim, action_dim, buffer_size, batch_size, actor_lr, critic_lr, alpha_lr, target_entropy,
                     gamma, tau, bl, br, device)

    train_sac_agent(agent, Max_ep, total_simulation_time)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    agent.save_model(folder_name)

"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""
import math
import operator
import carla
import numpy as np
from multiprocessing import Process
import subprocess
from P3VI.train import P3VIWrapper
from config import Config

from benchmark.rlagent import RLAgent
from ped_path_predictor.m2p3 import PathPredictor
from benchmark.risk.risk_aware_path import PathPlanner


def run_server():
    subprocess.run(['cd your path to isdespot && ./car {}'.format(
        Config.despot_port)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


class HyREAL(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False, agent='HyREAL'):
        super(HyREAL, self).__init__(world, carla_map, scenario)

        self.conn = conn
        self.eval_mode = eval_mode
        self.agent = agent
        if not self.eval_mode and not conn is None:
            p = Process(target=run_server)
            p.start()
            self.conn.establish_connection()
            m = self.conn.receive_message()
            print(m)  # RESET
        #self.ped_pred = PathPredictor("ped_path_predictor/_out/m2p3_289271.pth")
        #self.ped_pred.model.eval()
        self.ped_pred = P3VIWrapper("your path to p3vi",observed_frame_num=self.observed_frame_num,predicting_frame_num=self.predicting_frame_num)

        self.risk_path_planner = PathPlanner()

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(HyREAL, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal

    def get_speed_action(self, path, control):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y

        if self.prev_action is not None:
            reward, goal, hit, near_miss, terminal = self.get_reward_despot(self.prev_des_action)
            terminal = goal or hit
        else:
            # handling first instance
            reward = 0
            terminal = False
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        if len(path) == 0:
            control.brake = 0.6
            self.prev_des_action = 2
        elif not self.pedestrian_observable:
            control.throttle = 0.6
        else:
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            if m == "START":
                self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
                m = self.conn.receive_message()
            self.prev_des_action = 1
            if m[0] == '0':
                control.throttle = 0.6
                self.prev_des_action = 0
            elif m[0] == '2':
                control.brake = 0.6
                self.prev_des_action = 2

        self.prev_action = control
        return control

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # t = time.time()
        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        #if len(obstacles):
        #    (path, risk), intention = self.get_path_with_reasoning(start, end, obstacles)
        #else:
        #    (path, risk), intention = self.get_path_simple(start, end, obstacles)
        # print("time taken: ", time.time() - t)
        path = self.find_path(start, end, self.grid_cost, obstacles)
        intention = self.get_car_intention(obstacles, path, start)
        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.
        # print("Angle: ", control.steer)

        # Best speed action for the given path
        #if not self.eval_mode:
        #    control = self.get_speed_action(path, control)
        #print("Hi")
        self.prev_action = control
        velocity = self.vehicle.get_velocity()
        self.prev_speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5)
        risk = 0.0
        return control, intention, 0, self.pedestrian_observable
    

    def get_car_intention(self, obstacles, path, start):
        #print(path)
        #print(start)
        #print(self.world.semseg_sensor.array)
        #print(self.ped_history)
        ped = self.world.walker
        x,y = int(ped.get_location().x), int(ped.get_location().y)
        #print(x,y)
        self.past_trajectory.append(start)
        car_intention = self.world.semseg_sensor.array.copy()
        #print(car_intention[car_intention[x-2:x+2, y-2:y+2,0]])
        if len(path) == 0:
            return car_intention
        x, y = self.get_local_coordinates(path)
        car_intention[y, x, :] = 255.0  # overlay planned path on input with while line, color of pedestrian

        x, y = self.get_local_coordinates(self.past_trajectory)
        car_intention[y, x, :] = 0.0  # overlay past trajectory on input with black line

        if len(self.ped_history) >= self.observed_frame_num:
            ped_path = np.array(self.ped_history)
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            x,y = self.get_local_coordinates(pedestrian_path)
            car_intention[y,x,:] = np.array([220, 20, 60]) 
        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(car_intention, file)
        # car_intention = np.transpose(car_intention, (2, 0, 1))
        # assert car_intention.shape[0] == 3
        #if self.plot_intention:
        #    plt.imshow(car_intention)
        #    self.fig.savefig("cp.png",dpi=1200)
        #    self.fig.clear()
        #    print("Done")
        #    time.sleep(1)
        return car_intention

"""
Author: Dikshant Gupta
Time: 05.04.22 15:43
"""

import carla
import numpy as np
from multiprocessing import Process
import subprocess
from P3VI.model import P3VI
import matplotlib.pyplot as plt
from P3VI.train import P3VIWrapper
from config import Config
from benchmark.rlagent import RLAgent
import torch

from ped_path_predictor.m2p3 import PathPredictor

def run_server():
    subprocess.run(['cd your path to isdespot && ./car {}'.format(
        Config.despot_port)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


class ISDespotPStar(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(ISDespotPStar, self).__init__(world, carla_map, scenario)

        self.eval_mode = False
        self.conn = conn
        p = Process(target=run_server)
        p.start()
        self.conn.establish_connection()
        m = self.conn.receive_message()
        print(m)  # RESET
        if Config.pp == "p3vi":
            self.ped_pred = P3VIWrapper("your path to p3vi", 60, 80)

        else:
            self.ped_pred = PathPredictor("your path to m2p3")
            self.ped_pred.model.eval()
        #self.ped_pred.eval()
        self.pc = 0

    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(ISDespotPStar, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        #(path, risk), intention = self.get_path_simple(start, end, obstacles)
        path = self.find_path(start, end, self.grid_cost, obstacles)
        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, None, 0, self.pedestrian_observable

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
        elif False:#not self.pedestrian_observable: #Turn this off
            control.throttle = 0.6
        else:
            if len(self.ped_history) >= self.observed_frame_num:
                
                if Config.pp == "p3vi":
                    ped_path = np.array(self.ped_history)
                    #ped_path = ped_path.reshape((1,self.observed_frame_num, 4))
                    #print(ped_path)
                    pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
                else:
                    ped_path = np.array(self.ped_history)[:,0:2]

                    ped_path = ped_path.reshape((self.observed_frame_num, 2))
                    pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
                
                #
                    
                x,y = self.get_local_coordinates(pedestrian_path)
                intention = self.get_car_intention_plain()
                intention[y,x,:] = 255.0
                self.plot_intention = False
                pedestrian_path = pedestrian_path[::4]
                #print(len(pedestrian_path))
                if False:#self.plot_intention:
                    self.pc +=1
                    if self.pc % 5 == 0:
                        fig=plt.figure()
                        plt.imshow(intention)
                        fig.savefig("cp_debug/test_cp_%d.png"%self.pc,dpi=400)
            else:
                pedestrian_path = None
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed,
                                   pedestrian_positions, path, pedestrian_path)
            m = self.conn.receive_message()
            if m == "START":
                self.conn.send_message(terminal, reward, angle, car_pos, car_speed,
                                       pedestrian_positions, path, pedestrian_path)
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

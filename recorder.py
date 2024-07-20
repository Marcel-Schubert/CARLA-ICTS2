import os
import argparse
import subprocess
import time as t
import numpy as np
import sys
from datetime import datetime
from multiprocessing import Process
from SAC.sac_controller import SAC
import signal

from SAC.sac_discrete import SacdAgent
from SAC.sac_discrete.shared_sacd import SharedSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config
import time

file = "./SIMP3/data/01_int_test.npy"
def run(args):

    # Create environments.
    env = GIDASBenchmark(port=Config.port)
    #agent = SAC(env.world, env.map, env.scene)
    #env.reset_agent(agent)
    #test_env = GIDASBenchmark(port=Config.port + 100, setting="special")
    env.world.random = False
    env.extract = True
    data = []
    start_time = time.time()
    for i in range(Config.episodes):
        state = env.reset_extract()
        episode_length = 0
        ep_data = []
        print(i)
        #env.world.camera_manager.toggle_camera()
        #env.world.camera_manager.toggle_camera()
        #env.world.camera_manager.toggle_camera()
        env.world.camera_manager.toggle_recording()

        while episode_length < Config.max_episode_length:
            x,y,icr,son = env.record_step()
            ep_data.append((x,y,icr,son))
            episode_length+=1

        env.world.camera_manager.toggle_recording()
        break
        ep_data = np.array(ep_data)
        data.append(ep_data)

    env.close()




def run_server():
    # train environment
    port = "-carla-port={}".format(Config.port)
    if not Config.server:
        carla_p = "your path to carla"
        p = subprocess.run(['cd '+carla_p+' && ./CarlaUE4.sh your arguments' + port], shell=True)
        #cmd = 'cd '+carla_p+' && ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-server -benchmark -fps=50' + port
        #pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
        #                   shell=True, preexec_fn=os.setsid)
    else:
        carla_p = "your path to carla"
        command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
        p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
        
    return p


def run_test_server():
    # test environment
    port = "-carla-port={}".format(Config.port + 100)
    carla_p = "your path to carla"
    command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
    p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
    return p



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('SAC/sac_discrete/config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server', action='store_true')
    args = parser.parse_args()
    globals()["server"] = args.server
    Config.server = args.server

    Config.port = args.port
    print('Env. port: {}'.format(Config.port))

    p = Process(target=run_server)
    p.start()
    t.sleep(40)

    #p2 = Process(target=run_test_server)
    #p2.start()
    #time.sleep(5)
    run(args)
    subprocess.run(["kill -9 $(pidof CarlaUE4-Linux-Shipping)"], shell=True)

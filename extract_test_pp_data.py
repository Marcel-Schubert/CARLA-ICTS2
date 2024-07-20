import os
import yaml
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


def run(args):

    #file = "./P3VI/data/01_non_int_prelim.npy"
    file = "./P3VI/data/{}_test.npy".format(args.test)

    print(file)
    # Create environments.
    env = GIDASBenchmark(port=Config.port,mode="TESTING")
    #agent = SAC(env.world, env.map, env.scene)
    #env.reset_agent(agent)
    #test_env = GIDASBenchmark(port=Config.port + 100, setting="special")
    env.world.random = True
    env.extract = True
    data = []
    start_time = time.time()

    iterations = len(env.test_episodes)
    print(iterations)
    for i in range(iterations):
        state = env.reset_extract()
        episode_length = 0
        ep_data = []
        while episode_length < Config.max_episode_length:
            x,y,icr,son = env.extract_step()
            ep_data.append((x,y,icr,son))
            episode_length+=1
        ep_data = np.array(ep_data)
        data.append(ep_data)
        if i % 10 == 0:
            print("Episode:", i)
            print(time.time()-start_time)
        if i % 50 == 0 :
            save_data = np.array(data)
            np.save(file, save_data, allow_pickle=True)
            print("Saved",i)
            
    with open(file,'rb') as f:
        arr = np.load(f, allow_pickle=True)
        print(arr[0])
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
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument("--test",type=str)
    args = parser.parse_args()
    Config.server = args.server
    Config.port = args.port
    print('Env. port: {}'.format(Config.port))
    Config.port = args.port
    Config.qw = args.qw
    #if args.int:
    #    Config.scenarios = ["01_int","02_int","03_int"]
    #else:
    # Config.scenarios = ["01_non_int"]
    Config.scenarios = [args.test]
    print(Config.scenarios)
    # p = Process(target=run_server)
    # p.start()
    t.sleep(20)

    #p2 = Process(target=run_test_server)
    #p2.start()
    #time.sleep(5)
    run(args)
    #subprocess.run(["kill -9 $(pidof CarlaUE4-Linux-Shipping)"], shell=True)

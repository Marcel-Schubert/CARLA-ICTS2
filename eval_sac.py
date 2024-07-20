"""
Author: Dikshant Gupta
Time: 28.09.22 07:38
"""

import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process


from SAC.sac_discrete import EvalSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = GIDASBenchmark(port=Config.port,mode="TESTING")
    #env.eval(current_episode=args.episode)

    # Specify the directory to log.
    name = config["name"]
    config.pop("name",None)
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, 'eval', f'{name}-seed{args.seed}-{time}')
    path = "your path to sac"
    # Create the agent.
    agent = EvalSacdAgent(
        env=env, test_env=env, log_dir=log_dir, path=path, cuda=args.cuda, current_episode=0,
        seed=args.seed, agent="sacd", **config)
    agent.evaluate()



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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='sacd')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server', action='store_true')
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument("--test", type=str, default="01_int")
    args = parser.parse_args()
    globals()["server"] = args.server
    Config.server = args.server
    args.config = os.path.join('SAC/sac_discrete/config', args.config+".yaml")
    Config.port = args.port
    Config.qw = args.qw
    print('Env. port: {}'.format(Config.port))
    if args.test:
        if args.test == "all":
            # TODO PAGI: ADD SCENARIO HERE
            Config.scenarios = ['01_int','02_int','03_int', '04_int', '05_int', '01_non_int','02_non_int','03_non_int']
        else:
            Config.scenarios = [args.test]
    p = Process(target=run_server)
    p.start()
    time.sleep(20)
    #if Config.server:
    #    p2 = Process(target=run_test_server)
    #    p2.start()
    #    t.sleep(20)
    
    run(args)

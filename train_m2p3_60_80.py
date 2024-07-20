"""
Author: Dikshant Gupta
Time: 05.03.22 15:58
"""
import os
import signal

from ped_path_predictor.m2p3_60_80 import PathPredictor

if __name__ == "__main__":
    p = PathPredictor()
    p.train()
    os.kill(os.getppid(), signal.SIGHUP)
from typing import Union
from fastapi import FastAPI

# Standard library imports
import csv

# 3rd party imports
import numpy as np
import torch
import os.path
from tqdm import tqdm

# Module imports
from config import get_config
from utils import plot_training_history
from train import train
from inference import inference
from rl_env import StatesGenerator, get_benchmark_rewards
import json
config, _ = get_config()
app = FastAPI()


@app.get("/tasks")
def read_root():
    task_list, mask_list,agent_allocation,agent_stats,heuristics_stats=inference(config)
    try:
        idx=task_list.tolist().index(0)
        task_list=task_list[:idx]
        mask_list=mask_list[:idx]
    except:
        pass
    return {"task_list":task_list.tolist(),
            "mask_list":mask_list.tolist(),
            "agent_allocation":agent_allocation,
            "agent_stats":agent_stats,
            "heuristics_stats":heuristics_stats}


    # return {list(task_list[0])}
            # "mask_list": mask_list,
            # "agent_allocation":agent_allocation,}





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
    task_list, mask_list,agent_allocation,agent_stats=inference(config)
    # task_list=[t.item() for t in task_list]
    # mask_list=[t.item() for t in task_list]

    return {"msg":task_list.tolist(),
            "mask_list":mask_list.tolist(),
            "agent_allocation":agent_allocation,
            "agent_stats":agent_stats}


    # return {list(task_list[0])}
            # "mask_list": mask_list,
            # "agent_allocation":agent_allocation,}




